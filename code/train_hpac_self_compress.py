#!/usr/bin/env python3
"""Rate-distortion training for a self-compressing exact-integer HPAC."""

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
from hpac_self_compress import (
    bit_depth_histogram,
    enable_self_compression,
    estimated_model_bits,
    set_deployed_bit_depths,
    variable_weight_bits,
)


def residuals(tokens: torch.Tensor) -> torch.Tensor:
    output = tokens.clone()
    output[1:] = (tokens[1:] - tokens[:-1]) % 5
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--lr-exponent", type=float, default=2e-4)
    parser.add_argument("--lr-bits", type=float, default=0.1)
    parser.add_argument("--bit-eps", type=float, default=1e-3)
    parser.add_argument("--rate-lambda", type=float, default=1.0)
    parser.add_argument("--qat-fraction", type=float, default=0.25)
    parser.add_argument("--init-bits", type=float, default=8.0)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--patch", type=int, default=64)
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
    parser.add_argument("--spm", action="store_true")
    parser.add_argument("--norm-gates", action="store_true")
    parser.add_argument(
        "--target-mode", choices=("raw", "residual"), default="raw"
    )
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

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
    enable_self_compression(model, args.init_bits)
    initial = torch.load(args.init, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(initial["state_dict"], strict=False)
    allowed_missing = {
        name for name in incompatible.missing_keys if name.endswith(".bit_depth")
    }
    unexpected_missing = set(incompatible.missing_keys) - allowed_missing
    if incompatible.unexpected_keys or unexpected_missing:
        raise ValueError(f"incompatible initialization checkpoint: {incompatible}")

    named_parameters = dict(model.named_parameters())
    bit_names = {name for name in named_parameters if name.endswith(".bit_depth")}
    exponent_names = {name for name in named_parameters if name.endswith(".exponent")}
    other_names = set(named_parameters) - bit_names - exponent_names
    groups = [
        {
            "params": [named_parameters[name] for name in sorted(other_names)],
            "lr": args.lr,
            "eps": 1e-8,
        },
        {
            "params": [named_parameters[name] for name in sorted(bit_names)],
            "lr": args.lr_bits,
            "eps": args.bit_eps,
            "weight_decay": 0.0,
        },
    ]
    if exponent_names:
        groups.append({
            "params": [named_parameters[name] for name in sorted(exponent_names)],
            "lr": args.lr_exponent,
            "eps": 1e-8,
        })
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.02
    )
    pixels = tokens.numel()
    frame_count = tokens.shape[0]

    @torch.no_grad()
    def evaluate() -> dict[str, object]:
        model.eval()
        set_deployed_bit_depths(model, True)
        nats = 0.0
        misses = 0
        for start in range(0, frame_count, args.eval_batch_size):
            end = min(start + args.eval_batch_size, frame_count)
            idx = torch.arange(start, end, device=device)
            target = tokens[start:end]
            logits = model(target, idx, previous[start:end])
            nats += float(F.cross_entropy(logits, target, reduction="sum"))
            misses += int((logits.argmax(dim=1) != target).sum().item())
        bpp = nats / math.log(2) / pixels
        model_bits = estimated_model_bits(model)
        token_bytes = math.ceil(bpp * pixels / 8)
        return {
            "bpp": bpp,
            "top1_error": misses / pixels,
            "estimated_token_bytes": token_bytes,
            "estimated_model_bytes": math.ceil(model_bits / 8),
            "estimated_joint_bytes": token_bytes + math.ceil(model_bits / 8),
            "bit_depth_histogram": bit_depth_histogram(model),
        }

    best = None
    history: list[dict[str, object]] = []
    started = time.time()
    initial_metrics = evaluate()
    print(json.dumps({"epoch": 0, **initial_metrics}), flush=True)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    qat_start = max(1, math.floor(args.epochs * (1.0 - args.qat_fraction)) + 1)
    for epoch in range(1, args.epochs + 1):
        model.train()
        discrete_bits = epoch >= qat_start
        set_deployed_bit_depths(model, discrete_bits)
        permutation = torch.randperm(frame_count, generator=generator, device=device)
        for start in range(0, frame_count, args.batch_size):
            idx = permutation[start:start + args.batch_size]
            target = tokens[idx]
            logits = model(target, idx, previous[idx])
            task_loss = F.cross_entropy(logits, target)
            rate_loss = (
                args.rate_lambda
                * math.log(2)
                * variable_weight_bits(model, deployed=False)
                / pixels
            )
            loss = task_loss + rate_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
        scheduler.step()
        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            metrics = evaluate()
            record = {
                "epoch": epoch,
                "phase": "discrete_qat" if discrete_bits else "continuous",
                **metrics,
                "elapsed_seconds": time.time() - started,
            }
            history.append(record)
            print(json.dumps(record), flush=True)
            if best is None or metrics["estimated_joint_bytes"] < best["estimated_joint_bytes"]:
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
        "estimated_token_bytes": best["estimated_token_bytes"],
        "estimated_model_bytes": best["estimated_model_bytes"],
        "estimated_joint_bytes": best["estimated_joint_bytes"],
        "bit_depth_histogram": best["bit_depth_histogram"],
        "history": history,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    args.save.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best["state_dict"], "result": result}, args.save)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
