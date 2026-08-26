#!/usr/bin/env python3
"""Fine-tune CPR1's int4 renderer after the discrete token grid is fixed."""

from __future__ import annotations

import argparse
import copy
import json
import lzma
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.func import functional_call

from learned_token_mvp import (
    add_recipe_imports,
    camera_and_seg_input,
    chunks,
    evaluate_exact,
    expected_flip_loss,
    load_deployed_submission,
    load_xz_torch,
    quantized_renderer_copy,
    render_frozen_slaves,
    renderer_from_assignments,
)


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
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=150)
    parser.add_argument("--renderer-lr", type=float, default=2e-7)
    parser.add_argument("--seg-weight", type=float, default=100.0)
    parser.add_argument("--pose-weight", type=float, default=1.0)
    parser.add_argument("--distill-weight", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.pairs, args.epochs, args.batch_size, args.eval_batch_size) < 1:
        raise ValueError("pairs, epochs, and batch sizes must be positive")
    if args.start_pair < 0 or args.start_pair + args.pairs > 600:
        raise ValueError("selected pair interval lies outside [0, 600)")
    if min(args.renderer_lr, args.eval_every) <= 0 or args.distill_weight < 0:
        raise ValueError("invalid learning rate, eval interval, or loss weight")

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
    from train_semantic_quantized import fake_quantize  # pylint: disable=import-error,import-outside-toplevel

    targets = (
        load_xz_torch(cache_path)
        if cache_path.suffix == ".xz"
        else torch.load(cache_path, map_location="cpu", weights_only=False)
    )
    pair_ids = list(range(args.start_pair, args.start_pair + args.pairs))
    baseline_tokens = targets["seg"][pair_ids].long()
    learned_tokens = load_token_grid(args.tokens, baseline_tokens.shape)
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
        basis, coeff, pair_ids, device, batch_size=args.eval_batch_size
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

    teacher_masters = []
    with torch.no_grad():
        for selected in chunks(list(range(args.pairs)), args.eval_batch_size):
            global_ids = torch.tensor(
                [pair_ids[index] for index in selected],
                dtype=torch.long,
                device=device,
            )
            teacher_masters.append(
                model(baseline_tokens[selected].to(device), global_ids)
                .to(torch.float16)
                .cpu()
            )
    teacher_masters = torch.cat(teacher_masters)

    wrapper = AssignmentRenderer(model)
    optimizer = torch.optim.Adam(renderer_parameters, lr=args.renderer_lr)
    steps_per_epoch = math.ceil(args.pairs / args.batch_size)
    total_steps = args.epochs * steps_per_epoch
    generator = torch.Generator().manual_seed(args.seed)
    order: list[int] = []
    cursor = 0
    best_state = copy.deepcopy(model.state_dict())
    best_metrics = initial
    history = []
    started = time.time()

    for step in range(1, total_steps + 1):
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
        master_camera, seg_input = camera_and_seg_input(master)
        seg_proxy = expected_flip_loss(
            segnet(seg_input), baseline_tokens[selected].to(device)
        )
        pose_pred = pose_output(
            posenet,
            torch.stack(
                [slaves[selected].to(device=device, dtype=torch.float32), master_camera],
                dim=1,
            ),
        )
        pose_mse = (
            pose_pred - pose_targets[selected].to(device)
        ).square().mean()
        distill = F.mse_loss(
            master / 255.0,
            teacher_masters[selected].to(device=device, dtype=torch.float32) / 255.0,
        )
        loss = (
            args.seg_weight * seg_proxy
            + args.pose_weight * torch.sqrt(10.0 * pose_mse + 1e-12)
            + args.distill_weight * distill
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % args.eval_every == 0 or step == total_steps:
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
            record = {
                "step": step,
                "epoch": step / steps_per_epoch,
                "elapsed_seconds": time.time() - started,
                "loss": float(loss.detach()),
                "seg_proxy": float(seg_proxy.detach()),
                "pose_mse_batch": float(pose_mse.detach()),
                "distill_mse_batch": float(distill.detach()),
                "perception_score_delta": (
                    metrics["semantic_pose_score_without_rate"]
                    - baseline["semantic_pose_score_without_rate"]
                ),
                "renderer_semantic_lzma9_bytes": len(
                    lzma.compress(semantic_blob, preset=9)
                ),
                **metrics,
            }
            history.append(record)
            print(json.dumps({"stage": "checkpoint", **record}), flush=True)
            if (
                metrics["semantic_pose_score_without_rate"]
                < best_metrics["semantic_pose_score_without_rate"]
            ):
                best_metrics = metrics
                best_state = copy.deepcopy(model.state_dict())

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
        "experiment": "fixed-token-grid-int4-renderer-hardening",
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "frozen_submission_sizes": frozen_sizes,
        "changed_tokens": int((learned_tokens != baseline_tokens).sum()),
        "baseline": baseline,
        "initial": initial,
        "learned_requantized": final_metrics,
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
