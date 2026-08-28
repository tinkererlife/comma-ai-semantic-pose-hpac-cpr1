from __future__ import annotations

import lzma
import numpy as np
import torch

from hpac_token_search import (
    TokenMove,
    accept_with_backtracking,
    projected_hpac_rate_score,
    quantized_probability_bits,
    rank_token_moves,
)
from materialize_learned_cache import replace_tokens
from replace_archive_tokens import replace_token_stream
from learned_token_mvp import (
    differentiable_rate_proxy,
    hard_conditional_entropy,
    local_logits_from_tokens,
    mask_token_gradients,
    projected_lzma_rate_score,
    propose_token_changes,
    replace_global_perception,
    semantic_pose_score,
    straight_through_one_hot,
    token_rate_statistics,
)


def test_straight_through_forward_is_hard_and_backward_is_soft() -> None:
    logits = torch.tensor([[[[0.1, 0.2, 1.0, -1.0, 0.0]]]], requires_grad=True)
    assignment, ids = straight_through_one_hot(logits, temperature=0.7)
    assert ids.item() == 2
    assert torch.equal(assignment.detach(), torch.tensor([[[[0, 0, 1, 0, 0]]]]).float())
    (assignment * torch.arange(5.0)).sum().backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad) == 5


def test_constant_grid_has_zero_context_entropy() -> None:
    assignments = torch.zeros((1, 4, 5, 5))
    assignments[..., 3] = 1
    assert torch.isclose(differentiable_rate_proxy(assignments), torch.tensor(0.0))
    hard = np.full((2, 4, 5), 3, dtype=np.uint8)
    entropy = hard_conditional_entropy(hard)
    assert entropy["horizontal_bits_per_token"] == 0.0
    assert entropy["vertical_bits_per_token"] == 0.0
    assert entropy["temporal_bits_per_token"] == 0.0


def test_rate_statistics_reflect_predictability() -> None:
    constant = torch.zeros((4, 32, 32), dtype=torch.uint8)
    noisy = torch.arange(constant.numel(), dtype=torch.int64).reshape_as(constant) % 5
    constant_rate = token_rate_statistics(constant)
    noisy_rate = token_rate_statistics(noisy)
    assert constant_rate["lzma9_bytes"] < noisy_rate["lzma9_bytes"]
    assert constant_rate["marginal_bits_per_token"] < noisy_rate["marginal_bits_per_token"]


def test_gradient_trust_region_keeps_only_best_pixels() -> None:
    parameter = torch.nn.Parameter(torch.zeros((2, 2, 5)))
    parameter.data[..., 0] = 0.25
    parameter.grad = torch.tensor(
        [
            [[4.0, -4.0, 0.0, 0.0, 0.0], [3.0, -3.0, 0.0, 0.0, 0.0]],
            [[2.0, -2.0, 0.0, 0.0, 0.0], [1.0, -1.0, 0.0, 0.0, 0.0]],
        ]
    )
    parameters = torch.nn.ParameterList([parameter])
    stats = mask_token_gradients(parameters, [0], max_pixels_per_frame=2)
    kept = parameter.grad.abs().sum(dim=-1) > 0
    assert torch.equal(kept, torch.tensor([[True, True], [False, False]]))
    assert stats["gradient_selected_pixels"] == 2


def test_gradient_trust_region_does_not_retry_attempted_pixel() -> None:
    parameter = torch.nn.Parameter(torch.zeros((1, 2, 5)))
    parameter.data[..., 0] = 0.25
    gradient = torch.tensor(
        [[[4.0, -4.0, 0.0, 0.0, 0.0], [2.0, -2.0, 0.0, 0.0, 0.0]]]
    )
    parameter.grad = gradient.clone()
    parameters = torch.nn.ParameterList([parameter])
    attempted = [torch.zeros((1, 2), dtype=torch.bool)]
    mask_token_gradients(parameters, [0], 1, attempted)
    assert torch.equal(attempted[0], torch.tensor([[True, False]]))

    parameter.grad = gradient.clone()
    mask_token_gradients(parameters, [0], 1, attempted)
    assert torch.equal(attempted[0], torch.tensor([[True, True]]))
    kept = parameter.grad.abs().sum(dim=-1) > 0
    assert torch.equal(kept, torch.tensor([[False, True]]))


def test_streaming_proposal_changes_best_pixel_to_best_category() -> None:
    tokens = torch.zeros((1, 2, 2), dtype=torch.long)
    logits = local_logits_from_tokens(tokens, margin=0.25, device=torch.device("cpu"))
    logits.grad = torch.tensor(
        [
            [
                [[4.0, -4.0, 0.0, 0.0, 0.0], [2.0, 0.0, -3.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0, -2.0, 0.0], [0.5, 0.0, 0.0, 0.0, -1.0]],
            ]
        ]
    )
    attempted = [torch.zeros((2, 2), dtype=torch.bool)]
    proposal, stats = propose_token_changes(logits, tokens, [0], 1, attempted)
    assert torch.equal(proposal, torch.tensor([[[1, 0], [0, 0]]]))
    assert torch.equal(attempted[0], torch.tensor([[True, False], [False, False]]))
    assert stats["gradient_selected_pixels"] == 1
    assert stats["gradient_positive_candidates"] == 4


def test_projected_rate_score_is_exact_for_full_window() -> None:
    expected = 25.0 * 191_052 / 37_545_489
    assert projected_lzma_rate_score(191_052, 600) == expected


def test_hpac_probability_bits_match_uniform_softmax() -> None:
    bits = quantized_probability_bits(torch.zeros((3, 5)))
    assert torch.allclose(bits, torch.full_like(bits, np.log2(5)), atol=1e-6)


def test_hpac_probability_bits_match_numpy_codec_math() -> None:
    logits = torch.tensor([
        [1.31, -2.19, 0.06, 4.0, -0.44],
        [-10.0, 2.24, 2.26, 1.0, 0.0],
    ])
    codes = np.clip(np.rint(logits.numpy() * 8), -32768, 32767).astype(np.int16)
    reference_logits = codes.astype(np.float64) / 8.0
    reference_logits -= reference_logits.max(axis=1, keepdims=True)
    probabilities = np.exp(reference_logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    probabilities = probabilities.astype(np.float32)
    expected = -np.log2(probabilities.astype(np.float64)).astype(np.float32)
    actual = quantized_probability_bits(logits)
    assert actual.device == logits.device
    assert torch.equal(actual, torch.from_numpy(expected))


def test_projected_hpac_bits_use_official_rate_coefficient() -> None:
    assert projected_hpac_rate_score(8 * 191_052, 600) == (
        25.0 * 191_052 / 37_545_489
    )


def test_rate_only_moves_are_spatially_separated() -> None:
    tokens = torch.zeros((1, 4, 4), dtype=torch.long)
    logits = local_logits_from_tokens(tokens, 0.25, torch.device("cpu"))
    logits.grad = torch.zeros_like(logits)
    costs = torch.full_like(logits, 10.0)
    costs[..., 0] = 5.0
    costs[..., 1] = 4.0
    costs[0, 0, 0, 1] = 0.0
    costs[0, 0, 1, 1] = 0.1
    costs[0, 3, 3, 1] = 0.2
    attempted = [torch.zeros((4, 4), dtype=torch.bool)]
    moves, stats = rank_token_moves(
        logits,
        tokens,
        [0],
        max_pixels_per_frame=2,
        attempted_masks=attempted,
        category_rate_bits=costs,
        rate_score_per_bit=1.0,
        minimum_distance=3,
        rate_only=True,
    )
    assert [(move.row, move.col, move.after) for move in moves] == [
        (0, 0, 1),
        (3, 3, 1),
    ]
    assert stats["proposal_mode"] == "rate"


def test_rate_only_retries_another_category_without_a_gradient() -> None:
    tokens = torch.zeros((1, 1, 1), dtype=torch.long)
    logits = local_logits_from_tokens(tokens, 0.25, torch.device("cpu")).detach()
    costs = torch.tensor([[[[5.0, 1.0, 2.0, 3.0, 4.0]]]])
    attempted = [torch.zeros((1, 1), dtype=torch.uint8)]

    first, _ = rank_token_moves(
        logits,
        tokens,
        [0],
        1,
        attempted_masks=attempted,
        category_rate_bits=costs,
        rate_score_per_bit=1.0,
        rate_only=True,
    )
    second, _ = rank_token_moves(
        logits,
        tokens,
        [0],
        1,
        attempted_masks=attempted,
        category_rate_bits=costs,
        rate_score_per_bit=1.0,
        rate_only=True,
    )
    assert [first[0].after, second[0].after] == [1, 2]
    assert int(attempted[0][0, 0]) == (1 << 1) | (1 << 2)


def test_global_perception_replaces_one_frame_before_pose_sqrt() -> None:
    before = {"segnet_distortion": 0.01, "posenet_distortion": 0.01}
    after = {"segnet_distortion": 0.02, "posenet_distortion": 0.04}
    seg, pose, score = replace_global_perception(
        global_seg=0.03,
        global_pose=0.04,
        local_before=before,
        local_after=after,
        changed_frames=1,
    )
    assert np.isclose(seg, 0.03 + 0.01 / 600)
    assert np.isclose(pose, 0.04 + 0.03 / 600)
    assert np.isclose(score, semantic_pose_score(seg, pose))


def test_backtracking_finds_useful_subset_of_rejected_batch() -> None:
    tokens = torch.zeros((1, 1, 4), dtype=torch.long)
    moves = [
        TokenMove(0, 0, 0, col, 0, 1, 1.0, 1.0, 0.0)
        for col in range(4)
    ]

    def objective(candidate: torch.Tensor):
        key = abs(int(candidate.sum()) - 1)
        return float(key), {"sum": int(candidate.sum())}

    result = accept_with_backtracking(
        tokens,
        moves,
        initial_key=1.0,
        initial_payload={"sum": 0},
        evaluate=objective,
        max_evaluations=10,
    )
    assert len(result.accepted_moves) == 1
    assert int(result.tokens.sum()) == 1
    assert result.key == 0.0
    assert result.rejected_batches >= 1


def test_replace_tokens_preserves_pose_and_counts_changes() -> None:
    cache = {
        "seg": torch.zeros((2, 2, 2), dtype=torch.uint8),
        "pose": torch.arange(12, dtype=torch.float32).reshape(2, 6),
    }
    output, changed = replace_tokens(cache, bytes([0, 1, 0, 0, 2, 0, 0, 0]))
    assert changed == 2
    assert output["seg"].tolist() == [[[0, 1], [0, 0]], [[2, 0], [0, 0]]]
    assert output["pose"] is cache["pose"]


def test_replace_tokens_can_overlay_a_partial_frame_window() -> None:
    cache = {
        "seg": torch.zeros((3, 2, 2), dtype=torch.uint8),
        "pose": torch.zeros((3, 6)),
    }
    output, changed = replace_tokens(cache, bytes([1, 0, 0, 2]), start_pair=1)
    assert changed == 2
    assert output["seg"].tolist() == [
        [[0, 0], [0, 0]],
        [[1, 0], [0, 2]],
        [[0, 0], [0, 0]],
    ]


def test_replace_token_stream_preserves_model_prefix(tmp_path) -> None:
    import struct
    import zipfile

    base = tmp_path / "base.zip"
    payload = struct.pack("<I", 3) + b"abc" + b"old!"
    with zipfile.ZipFile(base, "w") as archive:
        archive.writestr("p", payload)
    output = tmp_path / "learned.zip"
    report = replace_token_stream(base, b"new!data", output)
    with zipfile.ZipFile(output) as archive:
        rebuilt = archive.read("p")
    assert rebuilt[:7] == payload[:7]
    assert rebuilt[7:] == b"new!data"
    assert report["model_prefix_bytes"] == 7


def test_replace_semantic_preserves_carrier_and_hpac(tmp_path) -> None:
    import struct
    import zipfile

    base = tmp_path / "base.zip"
    old_semantic = b"old"
    carrier_and_hpac = b"carryhpac"
    models = struct.pack("<II", len(old_semantic), 5) + old_semantic + carrier_and_hpac
    compressed = lzma.compress(models)
    payload = struct.pack("<I", len(compressed)) + compressed + b"old!"
    with zipfile.ZipFile(base, "w") as archive:
        archive.writestr("p", payload)

    output = tmp_path / "learned.zip"
    report = replace_token_stream(base, b"new!", output, b"semantic")
    with zipfile.ZipFile(output) as archive:
        rebuilt = archive.read("p")
    model_size = struct.unpack_from("<I", rebuilt)[0]
    raw = lzma.decompress(rebuilt[4 : 4 + model_size])
    semantic_size, carrier_size = struct.unpack_from("<II", raw)
    assert semantic_size == len(b"semantic")
    assert carrier_size == 5
    assert raw[8 + semantic_size :] == carrier_and_hpac
    assert rebuilt[4 + model_size :] == b"new!"
    assert report["preserved_model_bytes"] == len(carrier_and_hpac)
