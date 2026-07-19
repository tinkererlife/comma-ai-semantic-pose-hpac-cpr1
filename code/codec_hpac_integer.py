#!/usr/bin/env python3
"""Arithmetic codec for exact-inference IntegerHPAC checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import constriction
import numpy as np
import torch

from codec_hpac_residual import H, N, NUM_CLASSES, W, group_masks
from hpac_integer import IntegerHPAC
from hpac_integer_sparse import SparseIntegerHPAC


def residuals(tokens: torch.Tensor) -> torch.Tensor:
    output = tokens.clone()
    output[1:] = (tokens[1:] - tokens[:-1]) % NUM_CLASSES
    return output


def probability_table(selected: torch.Tensor, digest) -> np.ndarray:
    codes = selected.mul(8).round().clamp(-32768, 32767).to(torch.int16)
    codes = codes.cpu().numpy()
    digest.update(codes.tobytes(order="C"))
    logits = codes.astype(np.float64) / 8
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities.astype(np.float32)


def load_model(path: Path, args, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = IntegerHPAC(
        channels=args.channels, patch=args.patch,
        delta=args.delta, frame_dim=args.frame_dim,
        norm_mode=args.norm_mode, activation=args.activation,
        use_frame_scale=args.frame_scale,
        weight_bound=args.weight_bound, activation_bound=args.activation_bound,
        use_weight_scales=args.weight_scales,
        weight_exponent_min=args.weight_exponent_min,
        use_spm=args.spm,
        use_norm_gates=args.norm_gates,
    )
    if args.self_compress:
        from hpac_self_compress import enable_self_compression, set_deployed_bit_depths

        enable_self_compression(model)
    model.load_state_dict(checkpoint["state_dict"])
    if args.self_compress:
        set_deployed_bit_depths(model, True)
    return model.eval().to(device)


@torch.no_grad()
def encode(model, raw_tokens, targets, masks, device, sparse=None):
    encoder = constriction.stream.queue.RangeEncoder()
    family = constriction.stream.model.Categorical(perfect=False)
    digest = hashlib.sha256()
    previous = torch.zeros((1, H, W), dtype=torch.long, device=device)
    ideal_bits = 0.0
    started = time.time()
    for frame in range(len(raw_tokens)):
        idx = torch.tensor([frame], dtype=torch.long, device=device)
        current = torch.zeros_like(previous)
        context = model.prepare_frame_context(idx, previous)
        target = targets[frame]
        for group, mask in enumerate(masks):
            if sparse is None:
                logits = model.cached_context_logits(current, context)
                selected = logits[0][:, mask].permute(1, 0).contiguous()
            else:
                selected = sparse.selected_logits(current, context, group)
            table = probability_table(selected, digest)
            symbols = target[mask].cpu().numpy().astype(np.int32)
            ideal_bits += float(-np.log2(
                table[np.arange(len(symbols)), symbols].astype(np.float64)
            ).sum())
            encoder.encode(symbols, family, table)
            current[0, mask] = target[mask]
        previous = raw_tokens[frame].view(1, H, W)
        if frame == 0 or (frame + 1) % 25 == 0:
            print(json.dumps({
                "encoded_frames": frame + 1,
                "elapsed_seconds": time.time() - started,
            }), flush=True)
    return encoder.get_compressed().tobytes(), digest.hexdigest(), ideal_bits


@torch.no_grad()
def decode(
    model, blob: bytes, frame_count: int, masks, device, target_mode: str,
    sparse=None,
):
    decoder = constriction.stream.queue.RangeDecoder(np.frombuffer(blob, dtype=np.uint32))
    family = constriction.stream.model.Categorical(perfect=False)
    digest = hashlib.sha256()
    output = torch.empty((frame_count, H, W), dtype=torch.uint8)
    previous = torch.zeros((1, H, W), dtype=torch.long, device=device)
    started = time.time()
    for frame in range(frame_count):
        idx = torch.tensor([frame], dtype=torch.long, device=device)
        current = torch.zeros_like(previous)
        context = model.prepare_frame_context(idx, previous)
        for group, mask in enumerate(masks):
            if sparse is None:
                logits = model.cached_context_logits(current, context)
                selected = logits[0][:, mask].permute(1, 0).contiguous()
            else:
                selected = sparse.selected_logits(current, context, group)
            table = probability_table(selected, digest)
            symbols = decoder.decode(family, table).astype(np.int64)
            current[0, mask] = torch.from_numpy(symbols).to(device)
        raw = current if target_mode == "raw" else (current + previous) % NUM_CLASSES
        previous = raw
        output[frame].copy_(raw[0].to(torch.uint8).cpu())
        if frame == 0 or (frame + 1) % 25 == 0:
            print(json.dumps({
                "decoded_frames": frame + 1,
                "elapsed_seconds": time.time() - started,
            }), flush=True)
    return output, digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
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
    parser.add_argument("--spm", action="store_true")
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--self-compress", action="store_true")
    parser.add_argument("--norm-gates", action="store_true")
    parser.add_argument(
        "--target-mode", choices=("raw", "residual"), default="residual"
    )
    parser.add_argument("--frames", type=int, default=N)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tokens-out", type=Path)
    parser.add_argument("--decode-from", type=Path)
    parser.add_argument("--raw-out", type=Path)
    parser.add_argument(
        "--require-exact",
        action="store_true",
        help="fail unless decoded tokens exactly match --cache",
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    model = load_model(args.checkpoint, args, device)
    masks = group_masks(args.patch, args.delta, device)
    sparse = SparseIntegerHPAC(model) if args.sparse else None
    if args.decode_from is not None:
        output, logit_hash = decode(
            model, args.decode_from.read_bytes(), args.frames, masks, device,
            args.target_mode, sparse,
        )
        raw = output.numpy().tobytes(order="C")
        result = {
            "frames": args.frames,
            "logit_hash_decode": logit_hash,
            "raw_token_sha256": hashlib.sha256(raw).hexdigest(),
        }
        if args.cache is not None:
            expected = torch.load(
                args.cache, map_location="cpu", weights_only=False
            )["seg"][:args.frames].to(torch.uint8)
            result["verified_exact"] = bool(torch.equal(output, expected))
            if args.require_exact and not result["verified_exact"]:
                raise RuntimeError(
                    "decoded tokens differ from the requested target cache"
                )
        elif args.require_exact:
            parser.error("--require-exact requires --cache")
        if args.raw_out is not None:
            args.raw_out.parent.mkdir(parents=True, exist_ok=True)
            args.raw_out.write_bytes(raw)
    else:
        if args.cache is None or args.tokens_out is None:
            parser.error("encoding requires --cache and --tokens-out")
        raw_tokens = torch.load(
            args.cache, map_location="cpu", weights_only=False
        )["seg"][:args.frames].long().to(device)
        targets = raw_tokens if args.target_mode == "raw" else residuals(raw_tokens)
        blob, logit_hash, ideal_bits = encode(
            model, raw_tokens, targets, masks, device, sparse
        )
        args.tokens_out.parent.mkdir(parents=True, exist_ok=True)
        args.tokens_out.write_bytes(blob)
        result = {
            "frames": args.frames,
            "token_bytes": len(blob),
            "token_bpp": len(blob) * 8 / targets.numel(),
            "ideal_bpp": ideal_bits / targets.numel(),
            "logit_hash_encode": logit_hash,
        }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
