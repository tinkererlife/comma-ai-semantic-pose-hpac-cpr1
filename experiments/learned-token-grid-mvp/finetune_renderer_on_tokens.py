#!/usr/bin/env python3
"""Fine-tune CPR1's int4 renderer after the discrete token grid is fixed."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import lzma
import math
import random
import struct
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.func import functional_call

from learned_token_mvp import (
    ORIGINAL_UNCOMPRESSED_BYTES,
    add_recipe_imports,
    camera_and_seg_input,
    chunks,
    evaluate_exact,
    load_deployed_submission,
    load_xz_torch,
    official_metric_predictions,
    quantized_renderer_copy,
    render_frozen_slaves,
    renderer_from_assignments,
)
from replace_archive_tokens import replace_token_stream


@dataclass(frozen=True)
class TrainingStage:
    """One monotonic curriculum step for the quantized renderer."""

    name: str
    epochs: int
    c1a_lambda: float
    c1a_sigma: float
    hard_pixel_boost: float


def _csv_values(raw: str, cast, label: str) -> list:
    try:
        values = [cast(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"invalid {label}: {raw}") from exc
    if not values:
        raise ValueError(f"{label} cannot be empty")
    return values


def build_training_stages(
    stage_epochs: str,
    c1a_lambdas: str,
    c1a_sigmas: str,
    hard_pixel_boost: float,
) -> list[TrainingStage]:
    epochs = _csv_values(stage_epochs, int, "stage epochs")
    lambdas = _csv_values(c1a_lambdas, float, "C1a lambdas")
    sigmas = _csv_values(c1a_sigmas, float, "C1a sigmas")
    if not (len(epochs) == len(lambdas) == len(sigmas)):
        raise ValueError("stage epochs, C1a lambdas, and C1a sigmas must align")
    if any(value < 1 for value in epochs):
        raise ValueError("every training stage needs at least one epoch")
    if any(value < 0 for value in lambdas):
        raise ValueError("C1a lambdas cannot be negative")
    if any(value <= 0 for value in sigmas):
        raise ValueError("C1a sigmas must be positive")
    if hard_pixel_boost < 0:
        raise ValueError("hard-pixel boost cannot be negative")
    return [
        TrainingStage(
            name=(
                "margin-qat"
                if index == 0 and c1a_lambda == 0
                else f"c1a-{index}"
            ),
            epochs=stage_epoch,
            c1a_lambda=c1a_lambda,
            c1a_sigma=c1a_sigma,
            hard_pixel_boost=(0.0 if c1a_lambda == 0 else hard_pixel_boost),
        )
        for index, (stage_epoch, c1a_lambda, c1a_sigma) in enumerate(
            zip(epochs, lambdas, sigmas, strict=True)
        )
    ]


def softplus_margin_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    tau: float,
    hard_margin: float,
    hard_pixel_boost: float,
) -> torch.Tensor:
    """Smoothed decision-margin loss with optional normalized hard-pixel focus."""
    if tau <= 0:
        raise ValueError("margin temperature must be positive")
    if hard_pixel_boost < 0:
        raise ValueError("hard-pixel boost cannot be negative")
    target_logit = logits.gather(1, target[:, None]).squeeze(1)
    target_mask = F.one_hot(target, logits.shape[1]).movedim(-1, 1).bool()
    strongest_other = logits.masked_fill(target_mask, -torch.inf).amax(dim=1)
    margin = target_logit - strongest_other
    losses = tau * F.softplus(-margin / tau)
    if hard_pixel_boost:
        weights = 1.0 + hard_pixel_boost * (margin.detach() < hard_margin)
        weights = weights / weights.mean().clamp_min(1e-12)
        losses = losses * weights
    return losses.mean()


def c1a_entropy(
    named_parameters: Iterable[tuple[str, torch.Tensor]],
    bits: int,
    sigma: float,
) -> torch.Tensor:
    """Soft entropy of the exact per-channel signed quantization alphabet.

    This is the renderer analogue of C1a: gradients shape floating weights so
    their eventual int4 codes have a compressible marginal distribution.
    """
    if not 2 <= bits <= 8:
        raise ValueError("quantization bits must lie in [2, 8]")
    if sigma <= 0:
        raise ValueError("C1a sigma must be positive")
    weighted_entropy = None
    weight_count = 0
    limit = (1 << (bits - 1)) - 1
    for name, value in named_parameters:
        if not value.requires_grad or value.ndim < 2:
            continue
        source = value.float()
        embedding = name.endswith("embed.weight")
        reduce_dims = (
            tuple(range(source.ndim - 1))
            if embedding
            else tuple(range(1, source.ndim))
        )
        scale = source.detach().abs().amax(
            dim=reduce_dims, keepdim=True
        ).clamp_min(1e-8) / limit
        normalized = (source / scale).clamp(-limit, limit)
        levels = torch.arange(
            -limit, limit + 1, device=source.device, dtype=source.dtype
        )
        assignments = torch.softmax(
            -0.5 * ((normalized[..., None] - levels) / sigma).square(), dim=-1
        )
        marginal = assignments.mean(dim=tuple(range(source.ndim)))
        entropy = -(
            marginal * marginal.clamp_min(1e-12).log2()
        ).sum()
        contribution = entropy * source.numel()
        weighted_entropy = (
            contribution
            if weighted_entropy is None
            else weighted_entropy + contribution
        )
        weight_count += source.numel()
    if weighted_entropy is None:
        raise ValueError("C1a received no trainable matrix parameters")
    return weighted_entropy / weight_count


def archive_token_stream(archive_path: Path) -> bytes:
    """Extract the existing range-coded tokens without decoding or changing them."""
    with zipfile.ZipFile(archive_path) as archive:
        payload = archive.read("p")
    if len(payload) < 4:
        raise ValueError("truncated CPR1 archive payload")
    model_bytes = struct.unpack_from("<I", payload)[0]
    offset = 4 + model_bytes
    if offset > len(payload):
        raise ValueError("CPR1 model prefix exceeds payload")
    return payload[offset:]


class AssignmentRenderer(torch.nn.Module):
    """Expose the renderer to functional_call for quantization-aware training."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, assignments: torch.Tensor, pair_ids: torch.Tensor
    ) -> torch.Tensor:
        return renderer_from_assignments(self.model, assignments, pair_ids)


def load_token_grid(path: Path, shape: torch.Size) -> torch.Tensor:
    raw = np.frombuffer(path.read_bytes(), dtype=np.uint8).copy()
    expected = math.prod(shape)
    if raw.size != expected:
        raise ValueError(f"token file has {raw.size} values, expected {expected}")
    if int(raw.max()) >= 5:
        raise ValueError("token IDs must lie in [0, 5)")
    return torch.from_numpy(raw).reshape(shape).long()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe-root", type=Path, required=True)
    parser.add_argument("--challenge-root", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--start-pair", type=int, default=0)
    parser.add_argument("--pairs", type=int, default=600)
    parser.add_argument(
        "--epochs",
        type=int,
        help="Legacy one-stage epoch count; overrides the staged curriculum.",
    )
    parser.add_argument(
        "--stage-epochs",
        default="1,1,1",
        help="Comma-separated epochs for margin-QAT and successive C1a stages.",
    )
    parser.add_argument(
        "--c1a-lambdas",
        default="0,0.002,0.01",
        help="Per-stage compression pressure; must align with --stage-epochs.",
    )
    parser.add_argument(
        "--c1a-sigmas",
        default="0.2,0.2,0.1",
        help="Per-stage soft int4 assignment widths.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=150)
    parser.add_argument("--renderer-lr", type=float, default=2e-7)
    parser.add_argument("--seg-weight", type=float, default=100.0)
    parser.add_argument("--pose-weight", type=float, default=1.0)
    parser.add_argument("--distill-weight", type=float, default=100.0)
    parser.add_argument("--margin-tau", type=float, default=0.3)
    parser.add_argument("--hard-margin", type=float, default=1.0)
    parser.add_argument("--hard-pixel-boost", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stages = (
        [TrainingStage("legacy-margin-qat", args.epochs, 0.0, 0.2, 0.0)]
        if args.epochs is not None
        else build_training_stages(
            args.stage_epochs,
            args.c1a_lambdas,
            args.c1a_sigmas,
            args.hard_pixel_boost,
        )
    )
    if any(stage.epochs < 1 for stage in stages):
        raise ValueError("every training stage needs at least one epoch")
    if min(args.pairs, args.batch_size, args.eval_batch_size) < 1:
        raise ValueError("pairs and batch sizes must be positive")
    if args.start_pair < 0 or args.start_pair + args.pairs > 600:
        raise ValueError("selected pair interval lies outside [0, 600)")
    if (
        min(args.renderer_lr, args.eval_every, args.margin_tau) <= 0
        or args.distill_weight < 0
    ):
        raise ValueError("invalid learning rate, eval interval, or loss weight")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        # Match inflate; official_metric_predictions enables evaluator TF32 locally.
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
    from train_semantic_quantized import fake_quantize  # pylint: disable=import-error,import-outside-toplevel

    args.out_dir.mkdir(parents=True, exist_ok=True)

    targets = (
        load_xz_torch(cache_path)
        if cache_path.suffix == ".xz"
        else torch.load(cache_path, map_location="cpu", weights_only=False)
    )
    pair_ids = list(range(args.start_pair, args.start_pair + args.pairs))
    baseline_tokens = targets["seg"][pair_ids].long()
    learned_tokens = load_token_grid(
        args.tokens, targets["seg"].shape
    )[pair_ids]
    pose_targets = targets["pose"][pair_ids].float()

    model, basis, coeff, frozen_sizes = load_deployed_submission(
        archive_path, device
    )
    model.eval()
    renderer_parameters = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not name.startswith("frame_embed."))
        if parameter.requires_grad:
            renderer_parameters.append(parameter)

    segnet = modules.SegNet().eval().to(device)
    segnet.load_state_dict(load_file(modules.segnet_sd_path, device=str(device)))
    posenet = modules.PoseNet().eval().to(device)
    posenet.load_state_dict(load_file(modules.posenet_sd_path, device=str(device)))
    for network in (segnet, posenet):
        for parameter in network.parameters():
            parameter.requires_grad_(False)

    slaves = render_frozen_slaves(
        basis, coeff, pair_ids, device, batch_size=64
    )
    baseline = evaluate_exact(
        model,
        baseline_tokens,
        pair_ids,
        slaves,
        baseline_tokens,
        pose_targets,
        segnet,
        posenet,
        pose_output,
        args.eval_batch_size,
        device,
    )
    initial = evaluate_exact(
        model,
        learned_tokens,
        pair_ids,
        slaves,
        baseline_tokens,
        pose_targets,
        segnet,
        posenet,
        pose_output,
        args.eval_batch_size,
        device,
    )
    print(json.dumps({"stage": "baseline", **baseline}), flush=True)
    print(json.dumps({"stage": "initial", **initial}), flush=True)

    token_stream = archive_token_stream(archive_path)
    initial_total_score = (
        initial["semantic_pose_score_without_rate"]
        + 25.0 * archive_path.stat().st_size / ORIGINAL_UNCOMPRESSED_BYTES
    )

    teacher_masters = None
    if args.distill_weight:
        teacher_batches = []
        with torch.no_grad():
            for selected in chunks(list(range(args.pairs)), args.eval_batch_size):
                global_ids = torch.tensor(
                    [pair_ids[index] for index in selected],
                    dtype=torch.long,
                    device=device,
                )
                teacher_batches.append(
                    model(baseline_tokens[selected].to(device), global_ids)
                    .to(torch.float16)
                    .cpu()
                )
        teacher_masters = torch.cat(teacher_batches)

    wrapper = AssignmentRenderer(model)
    steps_per_epoch = math.ceil(args.pairs / args.batch_size)
    total_steps = sum(stage.epochs for stage in stages) * steps_per_epoch
    generator = torch.Generator().manual_seed(args.seed)
    order: list[int] = []
    cursor = 0
    best_state = copy.deepcopy(model.state_dict())
    best_metrics = initial
    best_total_score = initial_total_score
    best_archive = archive_path.read_bytes()
    history = []
    started = time.time()
    global_step = 0
    completed_epochs = 0
    last_loss = torch.zeros((), device=device)
    last_seg_proxy = torch.zeros((), device=device)
    last_pose_mse = torch.zeros((), device=device)
    last_distill = torch.zeros((), device=device)
    last_c1a = torch.zeros((), device=device)

    for stage_index, stage in enumerate(stages):
        stage_steps = stage.epochs * steps_per_epoch
        optimizer = torch.optim.Adam(renderer_parameters, lr=args.renderer_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=stage_steps, eta_min=args.renderer_lr * 0.1
        )
        for stage_step in range(1, stage_steps + 1):
            global_step += 1
            if cursor >= len(order):
                order = torch.randperm(args.pairs, generator=generator).tolist()
                cursor = 0
            selected = order[cursor : cursor + args.batch_size]
            cursor += len(selected)
            global_ids = torch.tensor(
                [pair_ids[index] for index in selected],
                dtype=torch.long,
                device=device,
            )
            assignments = F.one_hot(
                learned_tokens[selected].to(device), 5
            ).to(torch.float32)
            parameters = {
                name: fake_quantize(value, 4, name.endswith("embed.weight"))
                for name, value in wrapper.named_parameters()
            }
            master = functional_call(wrapper, parameters, (assignments, global_ids))
            master_camera, _ = camera_and_seg_input(master)
            slave = slaves[selected].to(device=device, dtype=torch.float32)
            seg_logits, pose_pred = official_metric_predictions(
                segnet, posenet, slave, master_camera
            )
            seg_proxy = softplus_margin_loss(
                seg_logits,
                baseline_tokens[selected].to(device),
                args.margin_tau,
                args.hard_margin,
                stage.hard_pixel_boost,
            )
            pose_mse = (
                pose_pred - pose_targets[selected].to(device)
            ).square().mean()
            distill = (
                F.mse_loss(
                    master / 255.0,
                    teacher_masters[selected].to(
                        device=device, dtype=torch.float32
                    ) / 255.0,
                )
                if teacher_masters is not None
                else torch.zeros((), device=device)
            )
            rate_entropy = (
                c1a_entropy(wrapper.named_parameters(), 4, stage.c1a_sigma)
                if stage.c1a_lambda
                else torch.zeros((), device=device)
            )
            loss = (
                args.seg_weight * seg_proxy
                + args.pose_weight * torch.sqrt(10.0 * pose_mse + 1e-12)
                + args.distill_weight * distill
                + stage.c1a_lambda * rate_entropy
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            last_loss = loss
            last_seg_proxy = seg_proxy
            last_pose_mse = pose_mse
            last_distill = distill
            last_c1a = rate_entropy

            checkpoint_due = (
                global_step % args.eval_every == 0
                or stage_step == stage_steps
                or global_step == total_steps
            )
            if not checkpoint_due:
                continue

            quantized, semantic_blob = quantized_renderer_copy(model)
            quantized = quantized.to(device)
            metrics = evaluate_exact(
                quantized,
                learned_tokens,
                pair_ids,
                slaves,
                baseline_tokens,
                pose_targets,
                segnet,
                posenet,
                pose_output,
                args.eval_batch_size,
                device,
            )
            candidate_path = args.out_dir / "candidate-archive.zip"
            archive_report = replace_token_stream(
                archive_path,
                token_stream,
                candidate_path,
                semantic_blob,
            )
            total_score = (
                metrics["semantic_pose_score_without_rate"]
                + 25.0
                * archive_report["archive_bytes"]
                / ORIGINAL_UNCOMPRESSED_BYTES
            )
            record = {
                "stage_index": stage_index,
                "stage_name": stage.name,
                "stage_step": stage_step,
                "step": global_step,
                "epoch": completed_epochs + stage_step / steps_per_epoch,
                "elapsed_seconds": time.time() - started,
                "learning_rate": scheduler.get_last_lr()[0],
                "loss": float(last_loss.detach()),
                "seg_proxy": float(last_seg_proxy.detach()),
                "pose_mse_batch": float(last_pose_mse.detach()),
                "distill_mse_batch": float(last_distill.detach()),
                "c1a_entropy_bits": float(last_c1a.detach()),
                "c1a_lambda": stage.c1a_lambda,
                "c1a_sigma": stage.c1a_sigma,
                "hard_pixel_boost": stage.hard_pixel_boost,
                "perception_score_delta": (
                    metrics["semantic_pose_score_without_rate"]
                    - initial["semantic_pose_score_without_rate"]
                ),
                "renderer_semantic_lzma9_bytes": len(
                    lzma.compress(semantic_blob, preset=9)
                ),
                "archive_bytes": archive_report["archive_bytes"],
                "total_score": total_score,
                "total_score_delta": total_score - initial_total_score,
                **metrics,
            }
            history.append(record)
            print(json.dumps({"stage": "checkpoint", **record}), flush=True)
            if total_score < best_total_score:
                best_total_score = total_score
                best_metrics = metrics
                best_state = copy.deepcopy(model.state_dict())
                best_archive = candidate_path.read_bytes()
        completed_epochs += stage.epochs

    model.load_state_dict(best_state)
    final_model, semantic_blob = quantized_renderer_copy(model)
    final_model = final_model.to(device)
    final_metrics = evaluate_exact(
        final_model,
        learned_tokens,
        pair_ids,
        slaves,
        baseline_tokens,
        pose_targets,
        segnet,
        posenet,
        pose_output,
        args.eval_batch_size,
        device,
    )
    report = {
        "schema_version": 1,
        "experiment": "staged-margin-qat-c1a-renderer",
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "stages": [stage.__dict__ for stage in stages],
        "frozen_submission_sizes": frozen_sizes,
        "changed_tokens": int((learned_tokens != baseline_tokens).sum()),
        "baseline": baseline,
        "initial": initial,
        "learned_requantized": final_metrics,
        "initial_archive_bytes": archive_path.stat().st_size,
        "best_archive_bytes": len(best_archive),
        "best_archive_sha256": hashlib.sha256(best_archive).hexdigest(),
        "initial_total_score": initial_total_score,
        "best_total_score": best_total_score,
        "best_total_score_delta": best_total_score - initial_total_score,
        "selected_checkpoint_metrics": best_metrics,
        "renderer_semantic_payload_bytes": len(semantic_blob),
        "renderer_semantic_lzma9_bytes": len(lzma.compress(semantic_blob, preset=9)),
        "history": history,
        "elapsed_seconds": time.time() - started,
        "max_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.out_dir / "semantic.bin").write_bytes(semantic_blob)
    (args.out_dir / "best-archive.zip").write_bytes(best_archive)
    torch.save(
        {
            "state_dict": {key: value.cpu() for key, value in best_state.items()},
            "quant_bits": 4,
            "report": report,
        },
        args.out_dir / "renderer.pt",
    )
    print(json.dumps({"stage": "complete", **report}), flush=True)


if __name__ == "__main__":
    main()
