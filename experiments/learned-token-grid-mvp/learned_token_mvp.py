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
    parser.add_argument("--start-pair", type=int, default=0)
    parser.add_argument("--pairs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=1)
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
        "--accept-exact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep a proposal only if exact perception score plus rate improves.",
    )
    parser.add_argument("--seg-weight", type=float, default=100.0)
    parser.add_argument("--pose-weight", type=float, default=1.0)
    parser.add_argument("--rate-lambda", type=float, default=0.01)
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
) -> dict[str, float]:
    model.eval()
    mismatches = 0
    pixels = 0
    pose_error = 0.0
    samples = 0
    local_ids = list(range(len(pair_ids)))
    for selected in chunks(local_ids, batch_size):
        global_ids = torch.tensor(
            [pair_ids[index] for index in selected], dtype=torch.long, device=device
        )
        token_batch = hard_tokens[selected].to(device=device, dtype=torch.long)
        master_eval = model(token_batch, global_ids)
        master_camera, seg_input = camera_and_seg_input(master_eval)
        seg_pred = segnet(seg_input).argmax(dim=1)
        target_seg = seg_targets[selected].to(device)
        mismatches += int((seg_pred != target_seg).sum())
        pixels += target_seg.numel()

        slave = slaves[selected].to(device=device, dtype=torch.float32)
        pose_pred = pose_output(posenet, torch.stack([slave, master_camera], dim=1))
        target_pose = pose_targets[selected].to(device)
        pose_error += float((pose_pred - target_pose).square().sum())
        samples += target_pose.numel()

    seg_distortion = mismatches / pixels
    pose_distortion = pose_error / samples
    semantic_pose_score = 100.0 * seg_distortion + math.sqrt(
        10.0 * pose_distortion
    )
    return {
        "segnet_distortion": seg_distortion,
        "posenet_distortion": pose_distortion,
        "semantic_pose_score_without_rate": semantic_pose_score,
    }


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
    if args.accept_exact and args.batch_size != 1:
        raise ValueError("exact atomic acceptance currently requires --batch-size 1")
    if args.finetune_renderer:
        raise ValueError(
            "renderer fine-tuning belongs in the later joint-training experiment; "
            "this scalable gate is deliberately token-only"
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
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
    pair_ids = list(range(args.start_pair, args.start_pair + args.pairs))
    baseline_tokens = targets["seg"][pair_ids].long()
    seg_targets = baseline_tokens.clone()
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

    slaves = render_frozen_slaves(
        basis, coeff, pair_ids, device, batch_size=args.eval_batch_size
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
    )
    baseline_rate = token_rate_statistics(baseline_tokens)
    baseline_metrics["token_rate"] = baseline_rate
    print(json.dumps({"stage": "baseline", **baseline_metrics}), flush=True)

    generator = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(args.pairs, generator=generator).tolist()
    cursor = 0
    history: list[dict[str, object]] = []
    best_key = baseline_metrics[
        "semantic_pose_score_without_rate"
    ] + projected_lzma_rate_score(baseline_rate["lzma9_bytes"], args.pairs)
    current_tokens = baseline_tokens.clone()
    best_tokens = baseline_tokens.clone()
    best_state = copy.deepcopy(model.state_dict())
    attempted_masks = [
        torch.zeros((EVAL_H, EVAL_W), dtype=torch.bool)
        for _ in range(args.pairs)
    ]
    exact_frame_keys: list[float | None] = [None] * args.pairs
    accepted_proposals = 0
    rejected_proposals = 0
    started = time.time()

    for step in range(1, args.steps + 1):
        if cursor + args.batch_size > len(order):
            order = torch.randperm(args.pairs, generator=generator).tolist()
            cursor = 0
        selected = order[cursor : cursor + args.batch_size]
        cursor += args.batch_size
        progress = (step - 1) / max(args.steps - 1, 1)
        temperature = args.temperature_start * (
            args.temperature_end / args.temperature_start
        ) ** progress

        tokens_before = current_tokens[selected].clone()
        logits = local_logits_from_tokens(tokens_before, args.init_margin, device)
        assignments, _ = straight_through_one_hot(logits, temperature)
        global_ids = torch.tensor(
            [pair_ids[index] for index in selected], dtype=torch.long, device=device
        )
        master_eval = renderer_from_assignments(model, assignments, global_ids)
        master_camera, seg_input = camera_and_seg_input(master_eval)
        seg_logits = segnet(seg_input)
        target_seg = seg_targets[selected].to(device)
        seg_proxy = expected_flip_loss(seg_logits, target_seg)

        slave = slaves[selected].to(device=device, dtype=torch.float32)
        pose_pred = pose_output(posenet, torch.stack([slave, master_camera], dim=1))
        target_pose = pose_targets[selected].to(device)
        pose_mse = (pose_pred - target_pose).square().mean()
        rate_proxy = differentiable_rate_proxy(assignments)
        loss = (
            args.seg_weight * seg_proxy
            + args.pose_weight * torch.sqrt(10.0 * pose_mse + 1e-12)
            + args.rate_lambda * rate_proxy
        )

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

        loss.backward()
        candidate_tokens, gradient_stats = propose_token_changes(
            logits,
            tokens_before,
            selected,
            args.max_token_pixels_per_frame,
            attempted_masks,
        )

        proposal_accepted = False
        candidate_key = None
        if args.accept_exact and gradient_stats["gradient_selected_pixels"]:
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
                accepted_proposals += 1
                current_tokens[selected] = candidate_tokens
                exact_frame_keys[selected[0]] = candidate_key
            else:
                rejected_proposals += 1
        elif gradient_stats["gradient_selected_pixels"]:
            current_tokens[selected] = candidate_tokens
            proposal_accepted = True
            accepted_proposals += 1

        if step % args.eval_every == 0 or step == args.steps:
            learned_tokens = current_tokens.clone()
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
            rate = token_rate_statistics(learned_tokens)
            changes = int((learned_tokens != baseline_tokens).sum())
            selection_key = exact[
                "semantic_pose_score_without_rate"
            ] + projected_lzma_rate_score(rate["lzma9_bytes"], args.pairs)
            if selection_key < best_key:
                best_key = selection_key
                best_tokens = learned_tokens.clone()
                best_state = copy.deepcopy(model.state_dict())
            record = {
                "step": step,
                "elapsed_seconds": time.time() - started,
                "temperature": temperature,
                "loss": float(loss.detach()),
                "seg_proxy": float(seg_proxy.detach()),
                "pose_mse_batch": float(pose_mse.detach()),
                "rate_proxy_batch": float(rate_proxy.detach()),
                **gradient_stats,
                "proposal_accepted": proposal_accepted,
                "candidate_exact_rate_key": candidate_key,
                "accepted_proposals": accepted_proposals,
                "rejected_proposals": rejected_proposals,
                "changed_tokens": changes,
                "changed_fraction": changes / baseline_tokens.numel(),
                **exact,
                "token_lzma9_bytes": rate["lzma9_bytes"],
                "token_spatial_bits_per_token": rate[
                    "mean_spatial_bits_per_token"
                ],
            }
            history.append(record)
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
    report = {
        "schema_version": 1,
        "experiment": "free-discrete-token-grid-mvp",
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
        "accepted_proposals": accepted_proposals,
        "rejected_proposals": rejected_proposals,
        "renderer_semantic_payload_bytes": len(semantic_blob),
        "renderer_semantic_lzma9_bytes": len(lzma.compress(semantic_blob, preset=9)),
        "projected_600_token_lzma_delta_bytes": projected_token_delta,
        "proxy_rate_score_delta": proxy_rate_score_delta,
        "proxy_total_score_delta_after_int4": (
            learned_metrics_quantized["semantic_pose_score_without_rate"]
            - baseline_metrics["semantic_pose_score_without_rate"]
            + proxy_rate_score_delta
        ),
        "history": history,
        "elapsed_seconds": time.time() - started,
        "max_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
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
