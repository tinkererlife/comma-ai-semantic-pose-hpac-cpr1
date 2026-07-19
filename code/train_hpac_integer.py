#!/usr/bin/env python3
"""Train the exact-integer HPAC student on the fixed semantic maps."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from hpac_integer import IntegerHPAC


N = 600


def residuals(tokens: torch.Tensor) -> torch.Tensor:
    output = tokens.clone()
    output[1:] = (tokens[1:] - tokens[:-1]) % 5
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--init", type=Path)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument(
        "--stop-after-epoch",
        type=int,
        help=(
            "stop early while retaining --epochs as the cosine-schedule horizon; "
            "used to reproduce historically selected intermediate checkpoints"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-2)
    parser.add_argument(
        "--lr-weight", type=float,
        help="learning rate for weight codes in dyadically scaled layers",
    )
    parser.add_argument("--lr-exponent", type=float, default=1e-3)
    parser.add_argument("--lr-frame-scale", type=float)
    parser.add_argument("--lr-spm", type=float)
    parser.add_argument("--lr-norm-gate", type=float)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--patch", type=int, default=32)
    parser.add_argument("--delta", type=int, default=2)
    parser.add_argument("--frame-dim", type=int, default=8)
    parser.add_argument(
        "--norm-mode", choices=("none", "center", "power"), default="none"
    )
    parser.add_argument(
        "--activation", choices=("relu", "leaky"), default="relu"
    )
    parser.add_argument("--frame-scale", action="store_true")
    parser.add_argument("--weight-bound", type=int, default=127)
    parser.add_argument("--activation-bound", type=int, default=127)
    parser.add_argument("--weight-scales", action="store_true")
    parser.add_argument("--weight-exponent-min", type=int, default=-6)
    parser.add_argument("--migration-exponent", type=int, default=-1)
    parser.add_argument("--spm", action="store_true")
    parser.add_argument("--norm-gates", action="store_true")
    parser.add_argument(
        "--target-mode", choices=("raw", "residual"), default="residual"
    )
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    stop_after = args.stop_after_epoch or args.epochs
    if not 1 <= stop_after <= args.epochs:
        raise ValueError("--stop-after-epoch must be in [1, --epochs]")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    raw_tokens = torch.load(
        args.cache, map_location="cpu", weights_only=False
    )["seg"].long().to(device)
    tokens = raw_tokens if args.target_mode == "raw" else residuals(raw_tokens)
    previous = torch.zeros_like(raw_tokens)
    previous[1:] = raw_tokens[:-1]
    model = IntegerHPAC(
        channels=args.channels,
        patch=args.patch,
        delta=args.delta,
        frame_dim=args.frame_dim,
        norm_mode=args.norm_mode,
        activation=args.activation,
        use_frame_scale=args.frame_scale,
        weight_bound=args.weight_bound,
        activation_bound=args.activation_bound,
        use_weight_scales=args.weight_scales,
        weight_exponent_min=args.weight_exponent_min,
        use_spm=args.spm,
        use_norm_gates=args.norm_gates,
    ).to(device)
    if args.init is not None:
        initial = torch.load(args.init, map_location="cpu", weights_only=False)
        allow_missing = (
            args.frame_scale or args.weight_scales or args.spm
            or args.norm_gates
        )
        incompatible = model.load_state_dict(
            initial["state_dict"], strict=not allow_missing
        )
        unexpected_missing = {
            name for name in incompatible.missing_keys
            if not (
                name.startswith("frame_scale.") or name.endswith(".exponent")
                or name.startswith("spm_") or name.startswith("norm_")
            )
        }
        if incompatible.unexpected_keys or unexpected_missing:
            raise ValueError(
                f"incompatible initialization checkpoint: {incompatible}"
            )
        missing_exponents = {
            name for name in incompatible.missing_keys
            if name.endswith(".exponent")
        }
        if args.weight_scales and missing_exponents:
            if not args.weight_exponent_min <= args.migration_exponent <= 0:
                raise ValueError(
                    "migration exponent is outside the configured range"
                )
            multiplier = 1 << (-args.migration_exponent)
            with torch.no_grad():
                for module_name, module in model.named_modules():
                    if not hasattr(module, "exponent"):
                        continue
                    exponent_name = module_name + ".exponent"
                    if exponent_name not in missing_exponents:
                        continue
                    migrated = module.weight.round() * multiplier
                    if float(migrated.abs().max()) > module.weight_bound:
                        raise ValueError(
                            f"{module_name} cannot migrate to exponent "
                            f"{args.migration_exponent} within its weight bound"
                        )
                    module.weight.copy_(migrated)
                    module.exponent.fill_(args.migration_exponent)
    named_parameters = dict(model.named_parameters())
    exponent_names = {
        name for name in named_parameters if name.endswith(".exponent")
    }
    scaled_weight_names = {
        name[:-len("exponent")] + "weight" for name in exponent_names
    }
    frame_scale_names = {
        name for name in named_parameters if name.startswith("frame_scale.")
    }
    spm_names = {
        name for name in named_parameters if name.startswith("spm_")
    }
    norm_gate_names = {
        name for name in named_parameters if name.startswith("norm_")
    }
    scaled_weight_names -= frame_scale_names
    scaled_weight_names -= spm_names
    scaled_weight_names -= norm_gate_names
    exponent_names -= frame_scale_names
    exponent_names -= spm_names
    exponent_names -= norm_gate_names
    other_names = (
        set(named_parameters) - exponent_names - scaled_weight_names
        - frame_scale_names - spm_names - norm_gate_names
    )
    groups = [{
        "params": [
            named_parameters[name] for name in sorted(other_names)
        ],
        "lr": args.lr,
    }]
    if scaled_weight_names:
        groups.append({
            "params": [
                named_parameters[name] for name in sorted(scaled_weight_names)
            ],
            "lr": args.lr_weight if args.lr_weight is not None else args.lr * 8,
        })
    if exponent_names:
        groups.append({
            "params": [
                named_parameters[name] for name in sorted(exponent_names)
            ],
            "lr": args.lr_exponent,
        })
    if frame_scale_names:
        groups.append({
            "params": [
                named_parameters[name] for name in sorted(frame_scale_names)
            ],
            "lr": (
                args.lr_frame_scale
                if args.lr_frame_scale is not None else args.lr * 8
            ),
        })
    if spm_names:
        groups.append({
            "params": [named_parameters[name] for name in sorted(spm_names)],
            "lr": args.lr_spm if args.lr_spm is not None else args.lr * 4,
        })
    if norm_gate_names:
        groups.append({
            "params": [
                named_parameters[name] for name in sorted(norm_gate_names)
            ],
            "lr": (
                args.lr_norm_gate
                if args.lr_norm_gate is not None else args.lr * 8
            ),
        })
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.02
    )
    pixels = tokens.numel()

    @torch.no_grad()
    def evaluate():
        model.eval()
        nats = 0.0
        misses = 0
        for start in range(0, N, args.eval_batch_size):
            end = min(start + args.eval_batch_size, N)
            idx = torch.arange(start, end, device=device)
            target = tokens[start:end]
            logits = model(target, idx, previous[start:end])
            nats += float(F.cross_entropy(logits, target, reduction="sum"))
            misses += int((logits.argmax(dim=1) != target).sum().item())
        return {
            "bpp": nats / math.log(2) / pixels,
            "top1_error": misses / pixels,
        }

    best = None
    history = []
    started = time.time()
    print(json.dumps({"epoch": 0, **evaluate()}), flush=True)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    for epoch in range(1, stop_after + 1):
        model.train()
        permutation = torch.randperm(N, generator=generator, device=device)
        for start in range(0, N, args.batch_size):
            idx = permutation[start:start + args.batch_size]
            target = tokens[idx]
            logits = model(target, idx, previous[idx])
            loss = F.cross_entropy(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
        scheduler.step()
        if (
            epoch == 1
            or epoch % args.eval_every == 0
            or epoch == stop_after
        ):
            metrics = evaluate()
            record = {
                "epoch": epoch,
                **metrics,
                "estimated_token_bytes": math.ceil(
                    metrics["bpp"] * pixels / 8
                ),
                "elapsed_seconds": time.time() - started,
            }
            history.append(record)
            print(json.dumps(record), flush=True)
            if best is None or metrics["bpp"] < best["bpp"]:
                best = {
                    **metrics,
                    "epoch": epoch,
                    "state_dict": {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    },
                }
            args.save.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(),
                "config": vars(args),
                "history": history,
            }, args.save.with_name(args.save.stem + ".latest.pt"))

    result = {
        "best_epoch": best["epoch"],
        "best_bpp": best["bpp"],
        "best_top1_error": best["top1_error"],
        "estimated_token_bytes": math.ceil(best["bpp"] * pixels / 8),
        "history": history,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    args.save.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best["state_dict"],
        "result": result,
    }, args.save)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
