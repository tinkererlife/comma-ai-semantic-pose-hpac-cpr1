#!/usr/bin/env python3
"""Small A/B test for freely learned discrete token grids on CPR1 #130.

The frozen submission is the control.  The experiment keeps its five-symbol,
384x512 token interface, deployed semantic renderer, and pose carrier.  It
then treats the token at every selected grid position as a trainable categorical
variable.  The hard forward pass is discrete; a straight-through softmax
provides gradients.  Optionally, the shared renderer (but never its per-frame
embedding) can be fine-tuned as well.
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import lzma
import math
import random
import struct
import sys
import time
import zipfile
import zlib
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from hpac_token_search import (
    BacktrackResult,
    HPACRateOracle,
    TokenRegionMove,
    accept_with_backtracking,
    apply_token_moves,
    projected_hpac_rate_score,
    rank_token_moves,
    rank_token_regions,
)


N_TOTAL_PAIRS = 600
N_TOKENS = 5
EVAL_H, EVAL_W = 384, 512
CAMERA_H, CAMERA_W = 874, 1164
ORIGINAL_UNCOMPRESSED_BYTES = 37_545_489


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe-root", type=Path, required=True)
    parser.add_argument("--challenge-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument(
        "--initial-tokens",
        type=Path,
        help="Optional full-600 uint8 grid to improve instead of starting at #130.",
    )
    parser.add_argument("--start-pair", type=int, default=0)
    parser.add_argument("--pairs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--frame-order",
        choices=("random", "checkerboard"),
        default="random",
        help="Checkerboard batches contain only temporally independent frames.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=80)
    parser.add_argument("--token-lr", type=float, default=0.15)
    parser.add_argument("--init-margin", type=float, default=0.25)
    parser.add_argument("--temperature-start", type=float, default=1.0)
    parser.add_argument("--temperature-end", type=float, default=0.35)
    parser.add_argument(
        "--max-token-pixels-per-frame",
        type=int,
        default=1,
        help="Discrete trust region: only these most promising pixels update.",
    )
    parser.add_argument(
        "--proposal-candidates-per-frame",
        type=int,
        default=1,
        help=(
            "Independent token alternatives evaluated exactly per checkerboard "
            "frame; at most one can be accepted."
        ),
    )
    parser.add_argument(
        "--candidate-batch-size",
        type=int,
        default=60,
        help="Maximum independent Top-K alternatives in one localized HPAC batch.",
    )
    parser.add_argument(
        "--proposal-shapes",
        choices=("pixel", "runs", "multiscale", "large", "huge", "mega"),
        default="pixel",
        help="Exact candidate family: pixels or small constant-category regions.",
    )
    parser.add_argument(
        "--proposal-min-distance",
        type=int,
        default=0,
        help="Minimum Chebyshev distance between moves proposed together.",
    )
    parser.add_argument(
        "--backtrack-max-evals",
        type=int,
        default=1,
        help="Exact batch/split evaluations allowed per optimization step.",
    )
    parser.add_argument(
        "--proposal-mode",
        choices=("joint", "rate", "alternating"),
        default="joint",
        help="Rank by total gradient, HPAC rate alone, or alternate both.",
    )
    parser.add_argument(
        "--accept-exact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep a proposal only if exact perception score plus rate improves.",
    )
    parser.add_argument("--seg-weight", type=float, default=100.0)
    parser.add_argument("--pose-weight", type=float, default=1.0)
    parser.add_argument("--rate-lambda", type=float, default=0.01)
    parser.add_argument(
        "--rate-model",
        choices=("lzma", "hpac"),
        default="lzma",
        help="Use the legacy spatial/LZMA proxy or deployed #130 HPAC surprise.",
    )
    parser.add_argument(
        "--hpac-checkpoint",
        type=Path,
        help="Defaults to the frozen #130 self-compressing IntegerHPAC.",
    )
    parser.add_argument(
        "--initial-attempts",
        type=Path,
        help="Optional compressed category-attempt history from an earlier run.",
    )
    parser.add_argument("--finetune-renderer", action="store_true")
    parser.add_argument("--renderer-lr", type=float, default=2e-6)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def load_xz_torch(path: Path) -> dict[str, torch.Tensor]:
    with lzma.open(path, "rb") as stream:
        return torch.load(
            io.BytesIO(stream.read()), map_location="cpu", weights_only=False
        )


def add_recipe_imports(recipe_root: Path, challenge_root: Path) -> None:
    sys.path.insert(0, str(recipe_root / "code"))
    sys.path.insert(0, str(challenge_root))


def load_deployed_submission(
    archive_path: Path, device: torch.device
) -> tuple[torch.nn.Module, torch.Tensor, torch.Tensor, dict[str, int]]:
    # Imported after the caller installs the pinned recipe/challenge paths.
    import inflate  # pylint: disable=import-error,import-outside-toplevel

    with zipfile.ZipFile(archive_path) as archive:
        payload = archive.read("p")
    models_bytes = struct.unpack_from("<I", payload)[0]
    models_raw = lzma.decompress(payload[4 : 4 + models_bytes])
    semantic_bytes, carrier_bytes = struct.unpack_from("<II", models_raw)
    semantic_pose_bytes = 8 + semantic_bytes + carrier_bytes
    semantic, basis, coeff = inflate.unpack_semantic_pose(
        models_raw[:semantic_pose_bytes]
    )
    return (
        semantic.to(device),
        basis.float(),
        coeff.float(),
        {
            "semantic_bytes": semantic_bytes,
            "carrier_bytes": carrier_bytes,
            "models_compressed_bytes": models_bytes,
        },
    )


def normalized_basis(raw_basis: torch.Tensor) -> torch.Tensor:
    value = F.interpolate(
        raw_basis, size=(EVAL_H, EVAL_W), mode="bicubic", align_corners=False
    )
    value = value - value.mean(dim=(1, 2, 3), keepdim=True)
    rms = value.square().mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(1e-5)
    return value / rms


@torch.no_grad()
def render_frozen_slaves(
    basis: torch.Tensor,
    coeff: torch.Tensor,
    pair_ids: list[int],
    device: torch.device,
    batch_size: int = 4,
) -> torch.Tensor:
    basis = normalized_basis(basis.to(device))
    rendered = []
    for selected_ids in chunks(pair_ids, batch_size):
        chosen = coeff[selected_ids].to(device)
        carrier = torch.einsum("bk,kchw->bchw", chosen, basis)
        carrier = carrier / math.sqrt(basis.shape[0])
        slave_eval = (127.5 + 64.0 * carrier).clamp(0.0, 255.0).round()
        slave = F.interpolate(
            slave_eval,
            size=(CAMERA_H, CAMERA_W),
            mode="bicubic",
            align_corners=False,
        ).clamp(0.0, 255.0).round()
        rendered.append(slave.to(torch.uint8).cpu())
    return torch.cat(rendered)


class _StraightThroughHard(torch.autograd.Function):
    @staticmethod
    def forward(ctx, soft: torch.Tensor, hard: torch.Tensor) -> torch.Tensor:
        del ctx
        return hard

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        del ctx
        return gradient, None


def straight_through_one_hot(
    logits: torch.Tensor, temperature: float
) -> tuple[torch.Tensor, torch.Tensor]:
    soft = F.softmax(logits / temperature, dim=-1)
    hard_ids = soft.argmax(dim=-1)
    hard = F.one_hot(hard_ids, N_TOKENS).to(soft.dtype)
    return _StraightThroughHard.apply(soft, hard), hard_ids


def renderer_from_assignments(
    model: torch.nn.Module,
    assignments: torch.Tensor,
    pair_idx: torch.Tensor,
) -> torch.Tensor:
    """Run the exact #130 renderer after replacing integer lookup by ST weights."""
    value = assignments @ model.token_embed.weight
    value = value.permute(0, 3, 1, 2)
    value = model.coord_mix(
        torch.cat(
            [
                value,
                model.coordinates(
                    value.shape[0], value.device, value.dtype
                ),
            ],
            dim=1,
        )
    )
    frame = model.frame_embed(pair_idx)
    for block in model.blocks:
        value = block(value, frame)
    return torch.sigmoid(model.head(F.gelu(value))) * 255.0


def ste_uint8(value: torch.Tensor) -> torch.Tensor:
    clipped = value.clamp(0.0, 255.0)
    return clipped + (clipped.round() - clipped).detach()


def camera_and_seg_input(master_eval: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    camera = ste_uint8(
        F.interpolate(
            master_eval,
            size=(CAMERA_H, CAMERA_W),
            mode="bilinear",
            align_corners=False,
        )
    )
    seg_input = F.interpolate(
        camera, size=(EVAL_H, EVAL_W), mode="bilinear", align_corners=False
    )
    return camera, seg_input


def official_metric_predictions(
    segnet: torch.nn.Module,
    posenet: torch.nn.Module,
    slave_camera: torch.Tensor, master_camera: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    pair_bhwc = torch.stack([slave_camera, master_camera], dim=1).permute(
        0, 1, 3, 4, 2
    ).contiguous()
    pair_chw = pair_bhwc.permute(0, 1, 4, 2, 3)
    precision_api = getattr(torch.backends.cudnn, "conv", None)
    use_new_api = precision_api is not None and hasattr(
        precision_api, "fp32_precision"
    )
    if use_new_api:
        previous_cudnn_tf32 = precision_api.fp32_precision
        precision_api.fp32_precision = "tf32"
    else:
        previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cudnn.allow_tf32 = True
    try:
        seg_logits = segnet(segnet.preprocess_input(pair_chw))
        pose = posenet(posenet.preprocess_input(pair_chw))["pose"][:, :6]
    finally:
        if use_new_api:
            precision_api.fp32_precision = previous_cudnn_tf32
        else:
            torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32
    return seg_logits, pose


def expected_flip_loss(
    logits: torch.Tensor, target: torch.Tensor, tau: float = 0.15
) -> torch.Tensor:
    target_logit = logits.gather(1, target[:, None]).squeeze(1)
    target_mask = F.one_hot(target, logits.shape[1]).movedim(-1, 1).bool()
    strongest_other = logits.masked_fill(target_mask, -torch.inf).amax(dim=1)
    margin = target_logit - strongest_other
    return torch.sigmoid(-margin / tau).mean()


def pair_conditional_entropy(
    previous: torch.Tensor, current: torch.Tensor, eps: float = 1e-9
) -> torch.Tensor:
    previous = previous.reshape(-1, previous.shape[-1])
    current = current.reshape(-1, current.shape[-1])
    joint = previous.T @ current
    joint = joint / joint.sum().clamp_min(eps)
    previous_marginal = joint.sum(dim=1)
    joint_entropy = -(joint * torch.log2(joint.clamp_min(eps))).sum()
    previous_entropy = -(
        previous_marginal * torch.log2(previous_marginal.clamp_min(eps))
    ).sum()
    return joint_entropy - previous_entropy


def differentiable_rate_proxy(assignments: torch.Tensor) -> torch.Tensor:
    horizontal = pair_conditional_entropy(
        assignments[:, :, :-1], assignments[:, :, 1:]
    )
    vertical = pair_conditional_entropy(
        assignments[:, :-1, :], assignments[:, 1:, :]
    )
    return 0.5 * (horizontal + vertical)


def hard_conditional_entropy(tokens: np.ndarray) -> dict[str, float]:
    def conditional(previous: np.ndarray, current: np.ndarray) -> float:
        joint = np.bincount(
            (previous.reshape(-1) * N_TOKENS + current.reshape(-1)),
            minlength=N_TOKENS * N_TOKENS,
        ).astype(np.float64).reshape(N_TOKENS, N_TOKENS)
        joint /= max(joint.sum(), 1.0)
        marginal = joint.sum(axis=1)

        def entropy(probabilities: np.ndarray) -> float:
            positive = probabilities[probabilities > 0]
            return float(-(positive * np.log2(positive)).sum())

        return entropy(joint) - entropy(marginal)

    horizontal = conditional(tokens[:, :, :-1], tokens[:, :, 1:])
    vertical = conditional(tokens[:, :-1, :], tokens[:, 1:, :])
    temporal = (
        conditional(tokens[:-1], tokens[1:]) if tokens.shape[0] > 1 else 0.0
    )
    return {
        "horizontal_bits_per_token": horizontal,
        "vertical_bits_per_token": vertical,
        "temporal_bits_per_token": temporal,
        "mean_spatial_bits_per_token": 0.5 * (horizontal + vertical),
    }


def token_rate_statistics(tokens: torch.Tensor) -> dict[str, object]:
    array = tokens.detach().cpu().numpy().astype(np.uint8, copy=False)
    raw = array.tobytes(order="C")
    histogram = np.bincount(array.reshape(-1), minlength=N_TOKENS)
    probabilities = histogram / histogram.sum()
    positive = probabilities[probabilities > 0]
    return {
        "raw_u8_bytes": len(raw),
        "zlib9_bytes": len(zlib.compress(raw, level=9)),
        "lzma9_bytes": len(lzma.compress(raw, preset=9)),
        "marginal_bits_per_token": float(
            -(positive * np.log2(positive)).sum()
        ),
        "histogram": histogram.tolist(),
        **hard_conditional_entropy(array),
    }


def pack_attempt_history(attempted_masks: list[torch.Tensor]) -> bytes:
    """Compress the per-pixel five-bit category history for exact resume."""
    array = torch.stack(attempted_masks).to(torch.uint8).numpy()
    return zlib.compress(array.tobytes(order="C"), level=1)


def unpack_attempt_history(
    blob: bytes, frames: int, height: int = EVAL_H, width: int = EVAL_W
) -> list[torch.Tensor]:
    raw = zlib.decompress(blob)
    expected = frames * height * width
    if len(raw) != expected:
        raise ValueError(
            f"attempt history has {len(raw)} entries, expected {expected}"
        )
    array = np.frombuffer(raw, dtype=np.uint8).copy().reshape(frames, height, width)
    if int(array.max(initial=0)) >= 1 << N_TOKENS:
        raise ValueError("attempt history contains bits outside the token alphabet")
    return [torch.from_numpy(frame) for frame in array]


def chunks(values: list[int], size: int) -> Iterable[list[int]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


@torch.no_grad()
def evaluate_exact(
    model: torch.nn.Module,
    hard_tokens: torch.Tensor,
    pair_ids: list[int],
    slaves: torch.Tensor,
    seg_targets: torch.Tensor,
    pose_targets: torch.Tensor,
    segnet: torch.nn.Module,
    posenet: torch.nn.Module,
    pose_output,
    batch_size: int,
    device: torch.device,
    return_per_frame: bool = False,
) -> dict[str, float | list[dict[str, float]]]:
    model.eval()
    mismatches = torch.zeros((), dtype=torch.int64, device=device)
    pixels = 0
    pose_error = torch.zeros((), dtype=torch.float64, device=device)
    samples = 0
    frame_mismatch_batches: list[torch.Tensor] = []
    frame_pose_error_batches: list[torch.Tensor] = []
    local_ids = list(range(len(pair_ids)))
    for selected in chunks(local_ids, batch_size):
        global_ids = torch.tensor(
            [pair_ids[index] for index in selected], dtype=torch.long, device=device
        )
        token_batch = hard_tokens[selected].to(device=device, dtype=torch.long)
        master_eval = model(token_batch, global_ids)
        master_camera, _ = camera_and_seg_input(master_eval)
        slave = slaves[selected].to(device=device, dtype=torch.float32)
        seg_logits, pose_pred = official_metric_predictions(
            segnet, posenet, slave, master_camera
        )
        seg_pred = seg_logits.argmax(dim=1)
        target_seg = seg_targets[selected].to(device)
        frame_mismatches = (seg_pred != target_seg).reshape(
            len(selected), -1
        ).sum(dim=1)
        mismatches += frame_mismatches.sum()
        pixels += target_seg.numel()

        target_pose = pose_targets[selected].to(device)
        frame_pose_error = (pose_pred - target_pose).square().reshape(
            len(selected), -1
        ).sum(dim=1)
        pose_error += frame_pose_error.double().sum()
        samples += target_pose.numel()

        if return_per_frame:
            frame_mismatch_batches.append(frame_mismatches)
            frame_pose_error_batches.append(frame_pose_error)

    seg_distortion = float(mismatches) / pixels
    pose_distortion = float(pose_error) / samples
    metrics: dict[str, float | list[dict[str, float]]] = {
        "segnet_distortion": seg_distortion,
        "posenet_distortion": pose_distortion,
        "semantic_pose_score_without_rate": semantic_pose_score(
            seg_distortion, pose_distortion
        ),
    }
    if return_per_frame:
        frame_pixels = seg_targets[0].numel()
        frame_samples = pose_targets[0].numel()
        frame_mismatches = torch.cat(frame_mismatch_batches).cpu().tolist()
        frame_pose_errors = torch.cat(frame_pose_error_batches).cpu().tolist()
        per_frame = []
        for mismatch, error in zip(frame_mismatches, frame_pose_errors):
            frame_seg = mismatch / frame_pixels
            frame_pose = error / frame_samples
            per_frame.append({
                "segnet_distortion": frame_seg,
                "posenet_distortion": frame_pose,
                "semantic_pose_score_without_rate": semantic_pose_score(
                    frame_seg, frame_pose
                ),
            })
        metrics["per_frame"] = per_frame
    return metrics


def initialize_logits(tokens: torch.Tensor, margin: float) -> torch.nn.ParameterList:
    parameters = []
    for frame in tokens:
        logits = torch.zeros((*frame.shape, N_TOKENS), dtype=torch.float32)
        logits.scatter_(-1, frame.long().unsqueeze(-1), margin)
        parameters.append(torch.nn.Parameter(logits))
    return torch.nn.ParameterList(parameters)


def local_logits_from_tokens(
    tokens: torch.Tensor, margin: float, device: torch.device
) -> torch.Tensor:
    """Create differentiable logits only for the frames in the current batch."""
    logits = torch.zeros(
        (*tokens.shape, N_TOKENS), dtype=torch.float32, device=device
    )
    logits.scatter_(-1, tokens.to(device).long().unsqueeze(-1), margin)
    return logits.requires_grad_()


def hard_tokens_from_parameters(parameters: torch.nn.ParameterList) -> torch.Tensor:
    return torch.stack([parameter.detach().argmax(dim=-1).cpu() for parameter in parameters])


def mask_token_gradients(
    parameters: torch.nn.ParameterList,
    selected: list[int],
    max_pixels_per_frame: int,
    attempted_masks: list[torch.Tensor] | None = None,
) -> dict[str, float]:
    """Keep gradients only where a category change has the best local evidence."""
    if max_pixels_per_frame < 1:
        raise ValueError("--max-token-pixels-per-frame must be positive")
    selected_pixels = 0
    positive_candidates = 0
    largest_benefit = 0.0
    for index in selected:
        parameter = parameters[index]
        gradient = parameter.grad
        if gradient is None:
            continue
        current = parameter.detach().argmax(dim=-1)
        current_gradient = gradient.gather(-1, current.unsqueeze(-1)).squeeze(-1)
        alternative_gradient = gradient.masked_fill(
            F.one_hot(current, N_TOKENS).bool(), torch.inf
        ).amin(dim=-1)
        benefit = current_gradient - alternative_gradient
        flat_benefit = benefit.reshape(-1)
        positive = flat_benefit > 0
        if attempted_masks is not None:
            positive = positive & ~attempted_masks[index].reshape(-1)
        positive_count = int(positive.sum())
        keep_count = min(max_pixels_per_frame, positive_count)
        mask = torch.zeros_like(flat_benefit, dtype=torch.bool)
        if keep_count:
            candidates = flat_benefit.masked_fill(~positive, -torch.inf)
            chosen = candidates.topk(keep_count, sorted=False).indices
            mask[chosen] = True
            if attempted_masks is not None:
                attempted_masks[index].reshape(-1)[chosen] = True
            largest_benefit = max(
                largest_benefit, float(candidates[chosen].max().detach())
            )
        gradient.mul_(mask.reshape(*benefit.shape, 1))
        selected_pixels += keep_count
        positive_candidates += positive_count
    return {
        "gradient_selected_pixels": selected_pixels,
        "gradient_positive_candidates": positive_candidates,
        "gradient_largest_benefit": largest_benefit,
    }


def propose_token_changes(
    logits: torch.Tensor,
    current_tokens: torch.Tensor,
    selected: list[int],
    max_pixels_per_frame: int,
    attempted_masks: list[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Turn the strongest first-order evidence into explicit hard proposals.

    Unlike the small-window prototype, this does not keep five floating-point
    logits and Adam state for every token in all 600 frames.  The logits are
    local scratch space; accepted hard tokens are the persistent latent state.
    """
    if max_pixels_per_frame < 1:
        raise ValueError("--max-token-pixels-per-frame must be positive")
    if logits.grad is None:
        raise ValueError("logits must have a gradient before proposing changes")
    if logits.shape[:-1] != current_tokens.shape:
        raise ValueError("logits and current_tokens shapes do not match")
    if len(selected) != logits.shape[0]:
        raise ValueError("selected indices do not match the batch size")

    gradient = logits.grad
    current_device = current_tokens.to(logits.device).long()
    current_gradient = gradient.gather(
        -1, current_device.unsqueeze(-1)
    ).squeeze(-1)
    alternative_gradient, alternative_token = gradient.masked_fill(
        F.one_hot(current_device, N_TOKENS).bool(), torch.inf
    ).min(dim=-1)
    benefit = current_gradient - alternative_gradient
    proposals = current_tokens.clone()
    selected_pixels = 0
    positive_candidates = 0
    largest_benefit = 0.0

    for local_index, parameter_index in enumerate(selected):
        flat_benefit = benefit[local_index].reshape(-1)
        positive = flat_benefit > 0
        if attempted_masks is not None:
            attempted = attempted_masks[parameter_index].reshape(-1).to(logits.device)
            positive = positive & ~attempted
        positive_count = int(positive.sum())
        keep_count = min(max_pixels_per_frame, positive_count)
        if not keep_count:
            continue

        candidates = flat_benefit.masked_fill(~positive, -torch.inf)
        chosen = candidates.topk(keep_count, sorted=False).indices
        chosen_cpu = chosen.detach().cpu()
        replacement = alternative_token[local_index].reshape(-1)[chosen].detach().cpu()
        proposals[local_index].reshape(-1)[chosen_cpu] = replacement
        if attempted_masks is not None:
            attempted_masks[parameter_index].reshape(-1)[chosen_cpu] = True
        largest_benefit = max(
            largest_benefit, float(candidates[chosen].max().detach())
        )
        selected_pixels += keep_count
        positive_candidates += positive_count

    return proposals, {
        "gradient_selected_pixels": selected_pixels,
        "gradient_positive_candidates": positive_candidates,
        "gradient_largest_benefit": largest_benefit,
    }


def projected_lzma_rate_score(lzma_bytes: int, frames: int) -> float:
    """Official rate coefficient with a selected window projected to 600."""
    return (
        25.0
        * lzma_bytes
        * N_TOTAL_PAIRS
        / (frames * ORIGINAL_UNCOMPRESSED_BYTES)
    )


def semantic_pose_score(seg_distortion: float, pose_distortion: float) -> float:
    return 100.0 * seg_distortion + math.sqrt(10.0 * pose_distortion)


def aggregate_exact_metrics(
    metrics: list[dict[str, float]],
) -> dict[str, float]:
    seg = sum(item["segnet_distortion"] for item in metrics) / len(metrics)
    pose = sum(item["posenet_distortion"] for item in metrics) / len(metrics)
    return {
        "segnet_distortion": seg,
        "posenet_distortion": pose,
        "semantic_pose_score_without_rate": semantic_pose_score(seg, pose),
    }


def replace_global_perception(
    global_seg: float,
    global_pose: float,
    local_before: dict[str, float],
    local_after: dict[str, float],
    changed_frames: int,
) -> tuple[float, float, float]:
    """Replace equal-sized frame contributions in the official 600-frame score."""
    weight = changed_frames / N_TOTAL_PAIRS
    candidate_seg = global_seg + weight * (
        local_after["segnet_distortion"] - local_before["segnet_distortion"]
    )
    candidate_pose = global_pose + weight * (
        local_after["posenet_distortion"] - local_before["posenet_distortion"]
    )
    return (
        candidate_seg,
        candidate_pose,
        semantic_pose_score(candidate_seg, candidate_pose),
    )


def quantized_renderer_copy(model: torch.nn.Module):
    from pack_semantic_pose import pack_semantic  # pylint: disable=import-error,import-outside-toplevel

    checkpoint = {
        "quant_bits": 4,
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
    }
    blob, restored = pack_semantic(checkpoint)
    quantized = copy.deepcopy(model).cpu()
    quantized.load_state_dict(restored)
    return quantized, blob


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.pairs < 1 or args.batch_size < 1:
        raise ValueError("steps, pairs, and batch size must be positive")
    if args.start_pair < 0 or args.start_pair + args.pairs > N_TOTAL_PAIRS:
        raise ValueError("selected pair interval lies outside [0, 600)")
    if args.init_margin <= 0 or args.temperature_start <= 0 or args.temperature_end <= 0:
        raise ValueError("margin and temperatures must be positive")
    if args.max_token_pixels_per_frame < 1:
        raise ValueError("--max-token-pixels-per-frame must be positive")
    if args.proposal_candidates_per_frame < 1:
        raise ValueError("--proposal-candidates-per-frame must be positive")
    if args.candidate_batch_size < 1:
        raise ValueError("--candidate-batch-size must be positive")
    if args.proposal_min_distance < 0:
        raise ValueError("--proposal-min-distance cannot be negative")
    if args.backtrack_max_evals < 1:
        raise ValueError("--backtrack-max-evals must be positive")
    checkerboard_batch = (
        args.frame_order == "checkerboard"
        and args.rate_model == "hpac"
        and args.proposal_mode in ("rate", "joint")
    )
    if args.accept_exact and args.batch_size != 1 and not checkerboard_batch:
        raise ValueError(
            "exact batches require rate-only HPAC with --frame-order checkerboard"
        )
    if checkerboard_batch:
        if args.max_token_pixels_per_frame != 1:
            raise ValueError("checkerboard batching currently requires one move per frame")
        color_counts = [
            sum(pair_id % 2 == color for pair_id in range(
                args.start_pair, args.start_pair + args.pairs
            ))
            for color in (0, 1)
        ]
        if any(count % args.batch_size for count in color_counts if count):
            raise ValueError("batch size must divide each checkerboard color")
    elif args.proposal_candidates_per_frame != 1 or args.proposal_shapes != "pixel":
        raise ValueError(
            "Top-K alternatives require exact HPAC checkerboard batching"
        )
    if args.proposal_shapes != "pixel" and args.proposal_mode != "rate":
        raise ValueError("structural candidates currently require rate proposals")
    if args.rate_model == "hpac" and not args.accept_exact:
        raise ValueError("the HPAC rate oracle requires --accept-exact")
    if args.finetune_renderer:
        raise ValueError(
            "renderer fine-tuning belongs in the later joint-training experiment; "
            "this scalable gate is deliberately token-only"
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        # Match the deployed inflate path; metric forwards enable official TF32 locally.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    recipe_root = args.recipe_root.resolve()
    challenge_root = args.challenge_root.resolve()
    cache_path = args.cache or (
        recipe_root / "artifacts/caches/gt_cache_600_official_ada.pt.xz"
    )
    archive_path = args.archive or recipe_root / "artifacts/final/archive.zip"
    add_recipe_imports(recipe_root, challenge_root)

    import modules  # pylint: disable=import-error,import-outside-toplevel
    from pose_basis_oracle import pose_output  # pylint: disable=import-error,import-outside-toplevel

    targets = load_xz_torch(cache_path) if cache_path.suffix == ".xz" else torch.load(
        cache_path, map_location="cpu", weights_only=False
    )
    all_target_tokens = targets["seg"].to(torch.uint8)
    all_initial_tokens = all_target_tokens.clone()
    if args.initial_tokens:
        raw_initial = np.frombuffer(
            args.initial_tokens.read_bytes(), dtype=np.uint8
        ).copy()
        if raw_initial.size != all_target_tokens.numel():
            raise ValueError(
                f"initial grid has {raw_initial.size} tokens, "
                f"expected {all_target_tokens.numel()}"
            )
        if int(raw_initial.max()) >= N_TOKENS:
            raise ValueError("initial token IDs must lie in [0, 5)")
        all_initial_tokens = torch.from_numpy(raw_initial).reshape_as(
            all_target_tokens
        )
    pair_ids = list(range(args.start_pair, args.start_pair + args.pairs))
    baseline_tokens = all_initial_tokens[pair_ids].long()
    seg_targets = all_target_tokens[pair_ids].long()
    pose_targets = targets["pose"][pair_ids].float()

    model, basis, coeff, frozen_sizes = load_deployed_submission(archive_path, device)
    for parameter in model.parameters():
        parameter.requires_grad_(args.finetune_renderer)
    # The 600-row frame embedding is already a latent channel.  Updating its
    # selected rows would make the token-grid comparison invalid.
    for parameter in model.frame_embed.parameters():
        parameter.requires_grad_(False)

    segnet = modules.SegNet().eval().to(device)
    segnet.load_state_dict(load_file(modules.segnet_sd_path, device=str(device)))
    posenet = modules.PoseNet().eval().to(device)
    posenet.load_state_dict(load_file(modules.posenet_sd_path, device=str(device)))
    for network in (segnet, posenet):
        for parameter in network.parameters():
            parameter.requires_grad_(False)

    hpac_oracle = None
    if args.rate_model == "hpac":
        hpac_checkpoint = args.hpac_checkpoint or (
            recipe_root
            / "artifacts/checkpoints/hpac_selfcompress_l1_fastbits_e60.pt"
        )
        hpac_oracle = HPACRateOracle(
            hpac_checkpoint,
            recipe_root / "code",
            all_initial_tokens,
            args.start_pair,
            device,
        )
        print(json.dumps({
            "stage": "hpac_oracle",
            "checkpoint": str(hpac_checkpoint),
            "proposal_mode": args.proposal_mode,
            "proposal_pixels_per_frame": args.max_token_pixels_per_frame,
            "proposal_candidates_per_frame": args.proposal_candidates_per_frame,
            "candidate_batch_size": args.candidate_batch_size,
            "proposal_shapes": args.proposal_shapes,
            "proposal_min_distance": args.proposal_min_distance,
            "backtrack_max_evals": args.backtrack_max_evals,
            "score_gate": (
                "global_official" if args.pairs == N_TOTAL_PAIRS else "local_projected"
            ),
            "attempt_history": "token_category",
        }), flush=True)

    slaves = render_frozen_slaves(
        basis, coeff, pair_ids, device, batch_size=64
    )

    baseline_metrics = evaluate_exact(
        model,
        baseline_tokens,
        pair_ids,
        slaves,
        seg_targets,
        pose_targets,
        segnet,
        posenet,
        pose_output,
        args.eval_batch_size,
        device,
        return_per_frame=args.rate_model == "hpac",
    )
    exact_frame_metrics = (
        baseline_metrics.pop("per_frame")
        if args.rate_model == "hpac"
        else [None] * args.pairs
    )
    baseline_rate = token_rate_statistics(baseline_tokens)
    baseline_metrics["token_rate"] = baseline_rate
    print(json.dumps({"stage": "baseline", **baseline_metrics}), flush=True)

    generator = torch.Generator().manual_seed(args.seed)
    def fresh_order() -> list[int]:
        if args.frame_order == "random":
            return torch.randperm(args.pairs, generator=generator).tolist()
        colors = []
        for color in (0, 1):
            members = [
                index for index, pair_id in enumerate(pair_ids)
                if pair_id % 2 == color
            ]
            permutation = torch.randperm(
                len(members), generator=generator
            ).tolist()
            colors.extend(members[index] for index in permutation)
        return colors

    order = fresh_order()
    cursor = 0
    history: list[dict[str, object]] = []
    best_key = (
        0.0
        if args.rate_model == "hpac"
        else baseline_metrics["semantic_pose_score_without_rate"]
        + projected_lzma_rate_score(baseline_rate["lzma9_bytes"], args.pairs)
    )
    current_tokens = baseline_tokens.clone()
    best_tokens = baseline_tokens.clone()
    best_state = copy.deepcopy(model.state_dict())
    current_hpac_delta_bits = 0.0
    best_hpac_delta_bits = 0.0
    attempted_masks = [
        torch.zeros(
            (EVAL_H, EVAL_W),
            dtype=torch.uint8 if args.rate_model == "hpac" else torch.bool,
        )
        for _ in range(args.pairs)
    ]
    if args.initial_attempts:
        if args.rate_model != "hpac":
            raise ValueError("category attempt history is only supported with HPAC")
        attempted_masks = unpack_attempt_history(
            args.initial_attempts.read_bytes(), args.pairs
        )
    global_score_gate = args.rate_model == "hpac" and args.pairs == N_TOTAL_PAIRS
    current_global_seg = float(baseline_metrics["segnet_distortion"])
    current_global_pose = float(baseline_metrics["posenet_distortion"])
    current_rate_frame_bits: dict[int, float] = {}
    exact_frame_keys: list[float | None] = [None] * args.pairs
    accepted_proposals = 0
    accepted_rate_saving_proposals = 0
    accepted_lossy_rate_proposals = 0
    rejected_proposals = 0
    proposed_moves = 0
    backtrack_evaluations = 0
    rejected_proposal_batches = 0
    localized_rate_rechecks = 0
    localized_current_patches = 0
    localized_next_patches = 0
    topk_candidate_evaluations = 0
    accepted_region_shapes: dict[str, int] = {}
    accepted_region_token_changes = 0
    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for step in range(1, args.steps + 1):
        if cursor + args.batch_size > len(order):
            order = fresh_order()
            cursor = 0
        selected = order[cursor : cursor + args.batch_size]
        cursor += args.batch_size
        progress = (step - 1) / max(args.steps - 1, 1)
        temperature = args.temperature_start * (
            args.temperature_end / args.temperature_start
        ) ** progress

        tokens_before = current_tokens[selected].clone()
        logits = local_logits_from_tokens(tokens_before, args.init_margin, device)
        rate_only = args.rate_model == "hpac" and (
            args.proposal_mode == "rate"
            or (args.proposal_mode == "alternating" and step % 2 == 0)
        )
        if rate_only:
            logits = logits.detach()
            loss_value = 0.0
            seg_proxy_value = 0.0
            pose_mse_value = 0.0
            rate_proxy_value = 0.0
        else:
            assignments, _ = straight_through_one_hot(logits, temperature)
            assignments.retain_grad()
            global_ids = torch.tensor(
                [pair_ids[index] for index in selected],
                dtype=torch.long,
                device=device,
            )
            master_eval = renderer_from_assignments(model, assignments, global_ids)
            master_camera, _ = camera_and_seg_input(master_eval)
            slave = slaves[selected].to(device=device, dtype=torch.float32)
            seg_logits, pose_pred = official_metric_predictions(
                segnet, posenet, slave, master_camera
            )
            target_seg = seg_targets[selected].to(device)
            seg_proxy = expected_flip_loss(seg_logits, target_seg)

            target_pose = pose_targets[selected].to(device)
            pose_mse = (pose_pred - target_pose).square().mean()
            rate_proxy = (
                differentiable_rate_proxy(assignments)
                if args.rate_model == "lzma"
                else torch.zeros((), device=device)
            )
            loss = (
                args.seg_weight * seg_proxy
                + args.pose_weight * torch.sqrt(10.0 * pose_mse + 1e-12)
                + (
                    args.rate_lambda * rate_proxy
                    if args.rate_model == "lzma"
                    else 0.0
                )
            )
            loss.backward()
            if assignments.grad is None:
                raise RuntimeError("missing token-assignment gradient")
            # Rank finite category switches in one-hot space.  The gradient of
            # softmax logits is distorted by temperature and initialization.
            logits.grad = assignments.grad.detach()
            loss_value = float(loss.detach())
            seg_proxy_value = float(seg_proxy.detach())
            pose_mse_value = float(pose_mse.detach())
            rate_proxy_value = float(rate_proxy.detach())
            del (
                assignments,
                global_ids,
                master_eval,
                master_camera,
                seg_logits,
                target_seg,
                seg_proxy,
                slave,
                pose_pred,
                target_pose,
                pose_mse,
                rate_proxy,
                loss,
            )
        proposal_accepted = False
        candidate_key = None
        step_backtrack_evaluations = 0
        step_rejected_batches = 0
        step_perception_delta = 0.0
        step_hpac_delta_bits = 0.0
        step_total_delta = 0.0

        if args.rate_model == "hpac":
            exact_before = aggregate_exact_metrics([
                exact_frame_metrics[index] for index in selected
            ])
            if len(selected) == 1:
                rate_before_bits, rate_tables, rate_before_frames = (
                    hpac_oracle.affected_frame_bits(
                        current_tokens,
                        selected,
                        return_cost_tables=True,
                        cached_frame_bits=current_rate_frame_bits,
                    )
                )
            else:
                if len({pair_ids[index] % 2 for index in selected}) != 1:
                    raise ValueError("checkerboard batch crossed temporal colors")
                rate_before_bits, rate_tables, rate_before_frames = (
                    hpac_oracle.proposal_batch(
                        current_tokens,
                        selected,
                        cached_frame_bits=current_rate_frame_bits,
                    )
                )
            current_rate_frame_bits.update(rate_before_frames)
            category_rate_bits = torch.stack(
                [rate_tables[index] for index in selected]
            )
            current_for_rate = tokens_before.to(device).long().unsqueeze(-1)
            rate_proxy_value = float(
                category_rate_bits.gather(-1, current_for_rate).mean()
            )
            independent_topk = checkerboard_batch and (
                args.proposal_candidates_per_frame > 1
                or args.proposal_shapes != "pixel"
            )
            if args.proposal_shapes == "pixel":
                moves, gradient_stats = rank_token_moves(
                    logits,
                    tokens_before,
                    selected,
                    (
                        args.proposal_candidates_per_frame
                        if independent_topk
                        else args.max_token_pixels_per_frame
                    ),
                    attempted_masks,
                    category_rate_bits=category_rate_bits,
                    rate_score_per_bit=projected_hpac_rate_score(
                        1.0, len(selected)
                    ),
                    minimum_distance=args.proposal_min_distance,
                    rate_only=rate_only,
                    independent_alternatives=independent_topk,
                )
            else:
                if args.proposal_shapes == "runs":
                    region_shapes = ((1, 2), (2, 1))
                elif args.proposal_shapes == "multiscale":
                    region_shapes = ((1, 2), (2, 1), (2, 2), (3, 3))
                elif args.proposal_shapes == "large":
                    region_shapes = ((1, 4), (4, 1), (3, 3), (4, 4), (5, 5))
                elif args.proposal_shapes == "huge":
                    region_shapes = ((1, 8), (8, 1), (5, 5), (7, 7), (9, 9))
                else:
                    region_shapes = ((1, 16), (16, 1), (9, 9), (13, 13), (17, 17))
                moves, gradient_stats = rank_token_regions(
                    tokens_before,
                    selected,
                    args.proposal_candidates_per_frame,
                    attempted_masks,
                    category_rate_bits,
                    region_shapes,
                    args.proposal_min_distance,
                )
            proposed_moves += len(moves)
            if global_score_gate:
                perception_before = semantic_pose_score(
                    current_global_seg, current_global_pose
                )
                rate_scale_frames = N_TOTAL_PAIRS
            else:
                perception_before = exact_before[
                    "semantic_pose_score_without_rate"
                ]
                rate_scale_frames = len(selected)
            key_before = perception_before + projected_hpac_rate_score(
                rate_before_bits, rate_scale_frames
            )
            initial_payload = {
                "exact": exact_before,
                "rate_bits": rate_before_bits,
                "frame_bits": rate_before_frames,
                "perception_score": perception_before,
                "global_seg": current_global_seg,
                "global_pose": current_global_pose,
            }

            def evaluate_candidate(candidate_tokens):
                nonlocal localized_rate_rechecks
                nonlocal localized_current_patches
                nonlocal localized_next_patches
                exact_candidate = evaluate_exact(
                    model,
                    candidate_tokens,
                    [pair_ids[index] for index in selected],
                    slaves[selected],
                    seg_targets[selected],
                    pose_targets[selected],
                    segnet,
                    posenet,
                    pose_output,
                    args.eval_batch_size,
                    device,
                    return_per_frame=True,
                )
                candidate_per_frame = exact_candidate.pop("per_frame")
                localized_delta_bits = 0.0
                localized_frame_deltas: dict[int, float] = {}
                individual_rate_deltas: dict[int, dict[str, object]] = {}
                changed_batch_indices = [
                    batch_index
                    for batch_index in range(len(selected))
                    if not torch.equal(
                        candidate_tokens[batch_index],
                        tokens_before[batch_index],
                    )
                ]
                if len(changed_batch_indices) == 1:
                    batch_index = changed_batch_indices[0]
                    localized_results = [hpac_oracle.localized_move_delta(
                        current_tokens,
                        selected[batch_index],
                        candidate_tokens[batch_index],
                    )]
                elif changed_batch_indices:
                    localized_results = hpac_oracle.localized_move_delta_batch(
                        current_tokens,
                        [selected[index] for index in changed_batch_indices],
                        torch.stack([
                            candidate_tokens[index]
                            for index in changed_batch_indices
                        ]),
                    )
                else:
                    localized_results = []
                for batch_index, localized_result in zip(
                    changed_batch_indices, localized_results
                ):
                    delta_bits, frame_deltas, locality = localized_result
                    localized_delta_bits += delta_bits
                    individual_rate_deltas[batch_index] = {
                        "bits": delta_bits,
                        "frame_deltas": frame_deltas,
                    }
                    for global_index, delta in frame_deltas.items():
                        localized_frame_deltas[global_index] = (
                            localized_frame_deltas.get(global_index, 0.0) + delta
                        )
                    localized_rate_rechecks += 1
                    localized_current_patches += locality["current_patches"]
                    localized_next_patches += locality["next_patches"]
                rate_candidate_bits = rate_before_bits + localized_delta_bits
                rate_candidate_frames = dict(rate_before_frames)
                for global_index, delta in localized_frame_deltas.items():
                    rate_candidate_frames[global_index] += delta
                if global_score_gate:
                    candidate_global_seg, candidate_global_pose, candidate_perception = (
                        replace_global_perception(
                            current_global_seg,
                            current_global_pose,
                            exact_before,
                            exact_candidate,
                            len(selected),
                        )
                    )
                else:
                    candidate_global_seg = current_global_seg
                    candidate_global_pose = current_global_pose
                    candidate_perception = exact_candidate[
                        "semantic_pose_score_without_rate"
                    ]
                unconstrained_key = candidate_perception + projected_hpac_rate_score(
                    rate_candidate_bits, rate_scale_frames
                )
                # A direct low-surprise symbol can still make later HPAC
                # contexts more expensive.  Pure rate mode requires the whole
                # affected stream to shrink before considering total score.
                rate_constraint_satisfied = (
                    not rate_only
                    or rate_candidate_bits < rate_before_bits
                )
                return (
                    unconstrained_key if rate_constraint_satisfied else math.inf
                ), {
                    "exact": exact_candidate,
                    "per_frame": candidate_per_frame,
                    "individual_rate_deltas": individual_rate_deltas,
                    "rate_bits": rate_candidate_bits,
                    "frame_bits": rate_candidate_frames,
                    "perception_score": candidate_perception,
                    "global_seg": candidate_global_seg,
                    "global_pose": candidate_global_pose,
                    "unconstrained_key": unconstrained_key,
                    "rate_constraint_satisfied": rate_constraint_satisfied,
                }

            if independent_topk:
                candidate_results: dict[
                    int, list[tuple[object, dict[str, float], float, dict[int, float]]]
                ] = {}
                for move_chunk in chunks(moves, args.candidate_batch_size):
                    candidate_frames = []
                    candidate_indices = []
                    candidate_pair_ids = []
                    for move in move_chunk:
                        candidate = tokens_before[move.batch_index].clone()
                        if isinstance(move, TokenRegionMove):
                            candidate[
                                move.row : move.row + move.height,
                                move.col : move.col + move.width,
                            ] = move.after
                        else:
                            if int(candidate[move.row, move.col]) != move.before:
                                raise ValueError(
                                    "Top-K move no longer matches source token"
                                )
                            candidate[move.row, move.col] = move.after
                        candidate_frames.append(candidate)
                        parameter_index = selected[move.batch_index]
                        candidate_indices.append(parameter_index)
                        candidate_pair_ids.append(pair_ids[parameter_index])
                    candidate_tensor = torch.stack(candidate_frames)
                    exact_candidates = evaluate_exact(
                        model,
                        candidate_tensor,
                        candidate_pair_ids,
                        slaves[candidate_indices],
                        seg_targets[candidate_indices],
                        pose_targets[candidate_indices],
                        segnet,
                        posenet,
                        pose_output,
                        args.eval_batch_size,
                        device,
                        return_per_frame=True,
                    )["per_frame"]
                    localized_results = hpac_oracle.localized_move_delta_batch(
                        current_tokens,
                        candidate_indices,
                        candidate_tensor,
                    )
                    for move, exact_after, localized_result in zip(
                        move_chunk, exact_candidates, localized_results
                    ):
                        rate_delta, frame_deltas, locality = localized_result
                        candidate_results.setdefault(move.batch_index, []).append((
                            move, exact_after, rate_delta, frame_deltas
                        ))
                        localized_rate_rechecks += 1
                        localized_current_patches += locality["current_patches"]
                        localized_next_patches += locality["next_patches"]
                        topk_candidate_evaluations += 1

                working_tokens = tokens_before.clone()
                accepted_moves = []
                accepted_per_frame = [
                    exact_frame_metrics[index] for index in selected
                ]
                accepted_frame_bits = dict(rate_before_frames)
                accepted_rate_delta = 0.0
                lossy_accepted = 0
                trial_global_seg = current_global_seg
                trial_global_pose = current_global_pose
                for batch_index, parameter_index in enumerate(selected):
                    exact_frame_before = exact_frame_metrics[parameter_index]
                    best_candidate = None
                    best_score_delta = 0.0
                    for move, exact_after, rate_delta, frame_deltas in (
                        candidate_results.get(batch_index, [])
                    ):
                        if rate_only and rate_delta >= 0:
                            continue
                        if global_score_gate:
                            perception_before_move = semantic_pose_score(
                                trial_global_seg, trial_global_pose
                            )
                            candidate_seg, candidate_pose, perception_after_move = (
                                replace_global_perception(
                                    trial_global_seg,
                                    trial_global_pose,
                                    exact_frame_before,
                                    exact_after,
                                    1,
                                )
                            )
                            score_delta = (
                                perception_after_move - perception_before_move
                                + projected_hpac_rate_score(
                                    rate_delta, N_TOTAL_PAIRS
                                )
                            )
                        else:
                            candidate_seg = trial_global_seg
                            candidate_pose = trial_global_pose
                            perception_before_move = exact_frame_before[
                                "semantic_pose_score_without_rate"
                            ]
                            perception_after_move = exact_after[
                                "semantic_pose_score_without_rate"
                            ]
                            score_delta = (
                                perception_after_move - perception_before_move
                                + projected_hpac_rate_score(rate_delta, 1)
                            )
                        if score_delta < best_score_delta:
                            best_score_delta = score_delta
                            best_candidate = (
                                move,
                                exact_after,
                                rate_delta,
                                frame_deltas,
                                candidate_seg,
                                candidate_pose,
                                perception_before_move,
                                perception_after_move,
                            )
                    if best_candidate is None:
                        continue
                    (
                        move,
                        exact_after,
                        rate_delta,
                        frame_deltas,
                        candidate_seg,
                        candidate_pose,
                        perception_before_move,
                        perception_after_move,
                    ) = best_candidate
                    if isinstance(move, TokenRegionMove):
                        working_tokens[
                            batch_index,
                            move.row : move.row + move.height,
                            move.col : move.col + move.width,
                        ] = move.after
                    else:
                        working_tokens[batch_index, move.row, move.col] = move.after
                    accepted_moves.append(move)
                    accepted_per_frame[batch_index] = exact_after
                    accepted_rate_delta += rate_delta
                    if perception_after_move > perception_before_move:
                        lossy_accepted += 1
                    for global_index, delta in frame_deltas.items():
                        accepted_frame_bits[global_index] += delta
                    if global_score_gate:
                        trial_global_seg = candidate_seg
                        trial_global_pose = candidate_pose

                accepted_exact = aggregate_exact_metrics(accepted_per_frame)
                accepted_perception = (
                    semantic_pose_score(trial_global_seg, trial_global_pose)
                    if global_score_gate
                    else accepted_exact["semantic_pose_score_without_rate"]
                )
                accepted_rate_bits = rate_before_bits + accepted_rate_delta
                result = BacktrackResult(
                    tokens=working_tokens,
                    key=accepted_perception + projected_hpac_rate_score(
                        accepted_rate_bits, rate_scale_frames
                    ),
                    payload={
                        "exact": accepted_exact,
                        "per_frame": accepted_per_frame,
                        "rate_bits": accepted_rate_bits,
                        "frame_bits": accepted_frame_bits,
                        "perception_score": accepted_perception,
                        "global_seg": trial_global_seg,
                        "global_pose": trial_global_pose,
                        "lossy_accepted_count": lossy_accepted,
                    },
                    accepted_moves=accepted_moves,
                    evaluations=len(moves),
                    rejected_batches=int(len(accepted_moves) != len(moves)),
                )
            elif checkerboard_batch and len(selected) > 1:
                all_candidate_tokens = apply_token_moves(tokens_before, moves)
                _, batch_payload = evaluate_candidate(all_candidate_tokens)
                move_by_batch = {move.batch_index: move for move in moves}
                working_tokens = tokens_before.clone()
                accepted_moves = []
                accepted_per_frame = [
                    exact_frame_metrics[index] for index in selected
                ]
                accepted_frame_bits = dict(rate_before_frames)
                accepted_rate_delta = 0.0
                lossy_accepted = 0
                trial_global_seg = current_global_seg
                trial_global_pose = current_global_pose
                for batch_index, parameter_index in enumerate(selected):
                    move = move_by_batch.get(batch_index)
                    if move is None:
                        continue
                    rate_details = batch_payload[
                        "individual_rate_deltas"
                    ][batch_index]
                    rate_delta = float(rate_details["bits"])
                    exact_after = batch_payload["per_frame"][batch_index]
                    exact_frame_before = exact_frame_metrics[parameter_index]
                    if global_score_gate:
                        perception_before_move = semantic_pose_score(
                            trial_global_seg, trial_global_pose
                        )
                        (
                            candidate_global_seg,
                            candidate_global_pose,
                            perception_after_move,
                        ) = replace_global_perception(
                            trial_global_seg,
                            trial_global_pose,
                            exact_frame_before,
                            exact_after,
                            1,
                        )
                        score_delta = (
                            perception_after_move - perception_before_move
                            + projected_hpac_rate_score(
                                rate_delta, N_TOTAL_PAIRS
                            )
                        )
                    else:
                        perception_before_move = exact_frame_before[
                            "semantic_pose_score_without_rate"
                        ]
                        perception_after_move = exact_after[
                            "semantic_pose_score_without_rate"
                        ]
                        score_delta = (
                            perception_after_move - perception_before_move
                            + projected_hpac_rate_score(rate_delta, 1)
                        )
                    if (rate_only and rate_delta >= 0) or score_delta >= 0:
                        continue

                    working_tokens[batch_index] = all_candidate_tokens[batch_index]
                    accepted_moves.append(move)
                    accepted_per_frame[batch_index] = exact_after
                    accepted_rate_delta += rate_delta
                    if perception_after_move > perception_before_move:
                        lossy_accepted += 1
                    for global_index, delta in rate_details[
                        "frame_deltas"
                    ].items():
                        accepted_frame_bits[global_index] += delta
                    if global_score_gate:
                        trial_global_seg = candidate_global_seg
                        trial_global_pose = candidate_global_pose

                accepted_exact = aggregate_exact_metrics(accepted_per_frame)
                accepted_perception = (
                    semantic_pose_score(trial_global_seg, trial_global_pose)
                    if global_score_gate
                    else accepted_exact["semantic_pose_score_without_rate"]
                )
                accepted_rate_bits = rate_before_bits + accepted_rate_delta
                accepted_key = accepted_perception + projected_hpac_rate_score(
                    accepted_rate_bits, rate_scale_frames
                )
                result = BacktrackResult(
                    tokens=working_tokens,
                    key=accepted_key,
                    payload={
                        "exact": accepted_exact,
                        "per_frame": accepted_per_frame,
                        "rate_bits": accepted_rate_bits,
                        "frame_bits": accepted_frame_bits,
                        "perception_score": accepted_perception,
                        "global_seg": trial_global_seg,
                        "global_pose": trial_global_pose,
                        "lossy_accepted_count": lossy_accepted,
                    },
                    accepted_moves=accepted_moves,
                    evaluations=1,
                    rejected_batches=int(len(accepted_moves) != len(moves)),
                )
            else:
                result = accept_with_backtracking(
                    tokens_before,
                    moves,
                    key_before,
                    initial_payload,
                    evaluate_candidate,
                    args.backtrack_max_evals,
                )
            accepted_count = len(result.accepted_moves)
            rejected_count = len(moves) - accepted_count
            accepted_proposals += accepted_count
            rejected_proposals += rejected_count
            backtrack_evaluations += result.evaluations
            rejected_proposal_batches += result.rejected_batches
            step_backtrack_evaluations = result.evaluations
            step_rejected_batches = result.rejected_batches
            candidate_key = result.key
            proposal_accepted = bool(accepted_count)
            if proposal_accepted:
                step_perception_delta = (
                    result.payload["perception_score"] - perception_before
                )
                step_hpac_delta_bits = (
                    float(result.payload["rate_bits"]) - rate_before_bits
                )
                step_total_delta = result.key - key_before
                if step_hpac_delta_bits < 0:
                    accepted_rate_saving_proposals += accepted_count
                    accepted_lossy_rate_proposals += result.payload.get(
                        "lossy_accepted_count",
                        accepted_count if step_perception_delta > 0 else 0,
                    )
                for move in result.accepted_moves:
                    if isinstance(move, TokenRegionMove):
                        shape = f"{move.height}x{move.width}"
                        accepted_region_shapes[shape] = (
                            accepted_region_shapes.get(shape, 0) + 1
                        )
                        accepted_region_token_changes += int((
                            tokens_before[
                                move.batch_index,
                                move.row : move.row + move.height,
                                move.col : move.col + move.width,
                            ] != move.after
                        ).sum())
                current_tokens[selected] = result.tokens
                current_hpac_delta_bits += step_hpac_delta_bits
                for batch_index, parameter_index in enumerate(selected):
                    exact_frame_metrics[parameter_index] = result.payload[
                        "per_frame"
                    ][batch_index]
                current_rate_frame_bits.update(result.payload["frame_bits"])
                if global_score_gate:
                    current_global_seg = result.payload["global_seg"]
                    current_global_pose = result.payload["global_pose"]
        else:
            if args.accept_exact:
                selected_index = selected[0]
                key_before = exact_frame_keys[selected_index]
                if key_before is None:
                    exact_before = evaluate_exact(
                        model,
                        tokens_before,
                        [pair_ids[selected_index]],
                        slaves[selected],
                        seg_targets[selected],
                        pose_targets[selected],
                        segnet,
                        posenet,
                        pose_output,
                        args.eval_batch_size,
                        device,
                    )
                    rate_before = token_rate_statistics(tokens_before)
                    key_before = exact_before[
                        "semantic_pose_score_without_rate"
                    ] + projected_lzma_rate_score(
                        rate_before["lzma9_bytes"], len(selected)
                    )
                    exact_frame_keys[selected_index] = key_before

            candidate_tokens, gradient_stats = propose_token_changes(
                logits,
                tokens_before,
                selected,
                args.max_token_pixels_per_frame,
                attempted_masks,
            )
            move_count = int(gradient_stats["gradient_selected_pixels"])
            proposed_moves += move_count
            if args.accept_exact and move_count:
                exact_candidate = evaluate_exact(
                    model,
                    candidate_tokens,
                    [pair_ids[index] for index in selected],
                    slaves[selected],
                    seg_targets[selected],
                    pose_targets[selected],
                    segnet,
                    posenet,
                    pose_output,
                    args.eval_batch_size,
                    device,
                )
                rate_candidate = token_rate_statistics(candidate_tokens)
                candidate_key = exact_candidate[
                    "semantic_pose_score_without_rate"
                ] + projected_lzma_rate_score(
                    rate_candidate["lzma9_bytes"], len(selected)
                )
                proposal_accepted = candidate_key < key_before
                if proposal_accepted:
                    accepted_proposals += move_count
                    current_tokens[selected] = candidate_tokens
                    exact_frame_keys[selected[0]] = candidate_key
                else:
                    rejected_proposals += move_count
            elif move_count:
                current_tokens[selected] = candidate_tokens
                proposal_accepted = True
                accepted_proposals += move_count

        if step % args.eval_every == 0 or step == args.steps:
            learned_tokens = current_tokens.clone()
            if global_score_gate:
                # Changed frames were already evaluated exactly.  Averaging the
                # cached metrics avoids rendering all 600 again every sweep.
                exact = aggregate_exact_metrics(exact_frame_metrics)
                current_global_seg = float(exact["segnet_distortion"])
                current_global_pose = float(exact["posenet_distortion"])
            else:
                exact = evaluate_exact(
                    model,
                    learned_tokens,
                    pair_ids,
                    slaves,
                    seg_targets,
                    pose_targets,
                    segnet,
                    posenet,
                    pose_output,
                    args.eval_batch_size,
                    device,
                )
            rate = (
                None
                if args.rate_model == "hpac"
                else token_rate_statistics(learned_tokens)
            )
            changes = int((learned_tokens != baseline_tokens).sum())
            if args.rate_model == "hpac":
                selection_key = (
                    exact["semantic_pose_score_without_rate"]
                    - baseline_metrics["semantic_pose_score_without_rate"]
                    + projected_hpac_rate_score(
                        current_hpac_delta_bits, args.pairs
                    )
                )
            else:
                selection_key = exact[
                    "semantic_pose_score_without_rate"
                ] + projected_lzma_rate_score(
                    rate["lzma9_bytes"], args.pairs  # type: ignore[index]
                )
            if selection_key < best_key:
                best_key = selection_key
                best_tokens = learned_tokens.clone()
                best_state = copy.deepcopy(model.state_dict())
                best_hpac_delta_bits = current_hpac_delta_bits
            record = {
                "step": step,
                "elapsed_seconds": time.time() - started,
                "temperature": temperature,
                "loss": loss_value,
                "seg_proxy": seg_proxy_value,
                "pose_mse_batch": pose_mse_value,
                "rate_proxy_batch": rate_proxy_value,
                **gradient_stats,
                "proposal_accepted": proposal_accepted,
                "candidate_exact_rate_key": candidate_key,
                "accepted_proposals": accepted_proposals,
                "rejected_proposals": rejected_proposals,
                "proposed_moves": proposed_moves,
                "backtrack_evaluations": backtrack_evaluations,
                "rejected_proposal_batches": rejected_proposal_batches,
                "step_backtrack_evaluations": step_backtrack_evaluations,
                "step_rejected_proposal_batches": step_rejected_batches,
                "step_perception_delta": step_perception_delta,
                "step_hpac_delta_bits": step_hpac_delta_bits,
                "step_total_score_delta": step_total_delta,
                "accepted_rate_saving_proposals": accepted_rate_saving_proposals,
                "accepted_lossy_rate_proposals": accepted_lossy_rate_proposals,
                "accepted_region_shapes": accepted_region_shapes,
                "accepted_region_token_changes": accepted_region_token_changes,
                "hpac_ideal_delta_bits": current_hpac_delta_bits,
                "hpac_rate_score_delta": projected_hpac_rate_score(
                    current_hpac_delta_bits, args.pairs
                ),
                "selection_key_delta": (
                    selection_key
                    if args.rate_model == "hpac"
                    else None
                ),
                "changed_tokens": changes,
                "changed_fraction": changes / baseline_tokens.numel(),
                **exact,
                "token_lzma9_bytes": (
                    None if rate is None else rate["lzma9_bytes"]
                ),
                "token_spatial_bits_per_token": (
                    None if rate is None else rate["mean_spatial_bits_per_token"]
                ),
            }
            history.append(record)
            (args.out_dir / "checkpoint_tokens.u8").write_bytes(
                learned_tokens.numpy().astype(np.uint8, copy=False).tobytes()
            )
            (args.out_dir / "checkpoint_attempts.u8.zlib").write_bytes(
                pack_attempt_history(attempted_masks)
            )
            (args.out_dir / "checkpoint.json").write_text(json.dumps({
                "schema_version": 2,
                "step": step,
                "current_hpac_delta_bits": current_hpac_delta_bits,
                "accepted_proposals": accepted_proposals,
                "accepted_lossy_rate_proposals": accepted_lossy_rate_proposals,
                "proposed_moves": proposed_moves,
                "topk_candidate_evaluations": topk_candidate_evaluations,
                "attempt_history_file": "checkpoint_attempts.u8.zlib",
                "last_record": record,
            }, indent=2) + "\n")
            print(json.dumps({"stage": "train", **record}), flush=True)

    model.load_state_dict(best_state)
    learned_metrics_float = evaluate_exact(
        model,
        best_tokens,
        pair_ids,
        slaves,
        seg_targets,
        pose_targets,
        segnet,
        posenet,
        pose_output,
        args.eval_batch_size,
        device,
    )
    quantized_model, semantic_blob = quantized_renderer_copy(model)
    quantized_model = quantized_model.to(device)
    learned_metrics_quantized = evaluate_exact(
        quantized_model,
        best_tokens,
        pair_ids,
        slaves,
        seg_targets,
        pose_targets,
        segnet,
        posenet,
        pose_output,
        args.eval_batch_size,
        device,
    )
    learned_rate = token_rate_statistics(best_tokens)
    changed_tokens = int((best_tokens != baseline_tokens).sum())
    projected_token_delta = round(
        (learned_rate["lzma9_bytes"] - baseline_rate["lzma9_bytes"])
        * N_TOTAL_PAIRS
        / args.pairs
    )
    proxy_rate_score_delta = (
        25.0 * projected_token_delta / ORIGINAL_UNCOMPRESSED_BYTES
    )
    hpac_rate_score_delta = projected_hpac_rate_score(
        best_hpac_delta_bits, args.pairs
    )
    active_rate_score_delta = (
        hpac_rate_score_delta
        if args.rate_model == "hpac"
        else proxy_rate_score_delta
    )
    report = {
        "schema_version": 2,
        "experiment": (
            "free-discrete-token-grid-v2-hpac"
            if args.rate_model == "hpac"
            else "free-discrete-token-grid-mvp"
        ),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "pair_ids": pair_ids,
        "frozen_submission_sizes": frozen_sizes,
        "baseline": baseline_metrics,
        "learned_float_renderer": {
            **learned_metrics_float,
            "token_rate": learned_rate,
        },
        "learned_requantized_int4_renderer": learned_metrics_quantized,
        "changed_tokens": changed_tokens,
        "changed_fraction": changed_tokens / baseline_tokens.numel(),
        "initial_token_changes_from_targets": int(
            (baseline_tokens != seg_targets).sum()
        ),
        "final_token_changes_from_targets": int(
            (best_tokens != seg_targets).sum()
        ),
        "accepted_proposals": accepted_proposals,
        "accepted_rate_saving_proposals": accepted_rate_saving_proposals,
        "accepted_lossy_rate_proposals": accepted_lossy_rate_proposals,
        "rejected_proposals": rejected_proposals,
        "proposed_moves": proposed_moves,
        "backtrack_evaluations": backtrack_evaluations,
        "rejected_proposal_batches": rejected_proposal_batches,
        "renderer_semantic_payload_bytes": len(semantic_blob),
        "renderer_semantic_lzma9_bytes": len(lzma.compress(semantic_blob, preset=9)),
        "projected_600_token_lzma_delta_bytes": projected_token_delta,
        "proxy_rate_score_delta": proxy_rate_score_delta,
        "hpac_ideal_delta_bits": best_hpac_delta_bits,
        "hpac_rate_score_delta": hpac_rate_score_delta,
        "active_rate_model": args.rate_model,
        "score_gate": (
            "global_official" if global_score_gate else "local_projected"
        ),
        "attempt_history": (
            "token_category" if args.rate_model == "hpac" else "pixel"
        ),
        "cached_hpac_frame_totals": len(current_rate_frame_bits),
        "localized_rate_rechecks": localized_rate_rechecks,
        "localized_current_patches": localized_current_patches,
        "localized_next_patches": localized_next_patches,
        "topk_candidate_evaluations": topk_candidate_evaluations,
        "accepted_region_shapes": accepted_region_shapes,
        "accepted_region_token_changes": accepted_region_token_changes,
        "proxy_total_score_delta_after_int4": (
            learned_metrics_quantized["semantic_pose_score_without_rate"]
            - baseline_metrics["semantic_pose_score_without_rate"]
            + active_rate_score_delta
        ),
        "history": history,
        "elapsed_seconds": time.time() - started,
        "max_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
    }
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.out_dir / "baseline_tokens.u8").write_bytes(
        baseline_tokens.numpy().astype(np.uint8, copy=False).tobytes()
    )
    (args.out_dir / "learned_tokens.u8").write_bytes(
        best_tokens.numpy().astype(np.uint8, copy=False).tobytes()
    )
    torch.save(
        {
            "state_dict": {key: value.cpu() for key, value in best_state.items()},
            "quant_bits": 4,
            "pair_ids": pair_ids,
            "report": report,
        },
        args.out_dir / "renderer.pt",
    )
    print(json.dumps({"stage": "complete", **report}, indent=2), flush=True)


if __name__ == "__main__":
    main()
