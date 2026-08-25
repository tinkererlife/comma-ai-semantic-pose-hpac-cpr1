from __future__ import annotations

import numpy as np
import torch

from materialize_learned_cache import replace_tokens
from replace_archive_tokens import replace_token_stream
from learned_token_mvp import (
    differentiable_rate_proxy,
    hard_conditional_entropy,
    local_logits_from_tokens,
    mask_token_gradients,
    projected_lzma_rate_score,
    propose_token_changes,
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


def test_replace_tokens_preserves_pose_and_counts_changes() -> None:
    cache = {
        "seg": torch.zeros((2, 2, 2), dtype=torch.uint8),
        "pose": torch.arange(12, dtype=torch.float32).reshape(2, 6),
    }
    output, changed = replace_tokens(cache, bytes([0, 1, 0, 0, 2, 0, 0, 0]))
    assert changed == 2
    assert output["seg"].tolist() == [[[0, 1], [0, 0]], [[2, 0], [0, 0]]]
    assert output["pose"] is cache["pose"]


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
