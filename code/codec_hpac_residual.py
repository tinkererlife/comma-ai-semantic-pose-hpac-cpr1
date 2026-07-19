#!/usr/bin/env python3
"""Deterministic vectorized HPAC residual-token encoder and verifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import constriction
import numpy as np
import torch
import torch.nn.functional as F

from pack_hpac_quantized import deserialize_packed, reconstruct_state


N = 600
H, W = 384, 512
NUM_CLASSES = 5


def residuals(tokens: torch.Tensor) -> torch.Tensor:
    result = torch.empty_like(tokens)
    result[0] = tokens[0]
    result[1:] = (tokens[1:] - tokens[:-1]) % NUM_CLASSES
    return result


def group_masks(patch: int, delta: int, device: torch.device) -> list[torch.Tensor]:
    rows = torch.arange(patch, device=device).view(patch, 1).expand(patch, patch)
    cols = torch.arange(patch, device=device).view(1, patch).expand(patch, patch)
    grid = cols + delta * rows
    nr, nc = H // patch, W // patch
    masks = []
    for group in range(int((1 + delta) * patch - delta)):
        local = grid == group
        full = local[None, None].expand(nr, nc, patch, patch)
        masks.append(full.permute(0, 2, 1, 3).reshape(H, W))
    return masks


def quantize_logits(
    selected_logits: torch.Tensor, precision: int, logit_reference: str,
    round_guard: float = 0.0,
) -> np.ndarray:
    if logit_reference == "class0":
        selected_logits = selected_logits - selected_logits[:, :1]
    elif logit_reference == "max":
        selected_logits = selected_logits - selected_logits.amax(dim=1, keepdim=True)
    scaled = selected_logits.mul(precision)
    quantized = scaled.round()
    if round_guard > 0:
        lower = scaled.floor()
        fraction = scaled - lower
        quantized = torch.where(
            (fraction - 0.5).abs() <= round_guard, lower, quantized
        )
    quantized = quantized.clamp(-32768, 32767).to(torch.int16)
    return quantized.cpu().numpy()


def probability_table(
    selected_logits: torch.Tensor,
    precision: int,
    digest,
    logit_reference: str,
    round_guard: float,
) -> np.ndarray:
    quantized_np = quantize_logits(
        selected_logits, precision, logit_reference, round_guard
    )
    digest.update(quantized_np.tobytes(order="C"))
    logits = quantized_np.astype(np.float64) / precision
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities.astype(np.float32)


def hierarchical_encode(encoder, families, table: np.ndarray, symbols: np.ndarray):
    top = table.argmax(axis=1).astype(np.int32)
    rows = np.arange(len(symbols))
    hit_probability = table[rows, top]
    binary_table = np.stack([1.0 - hit_probability, hit_probability], axis=1)
    hits = (symbols == top).astype(np.int32)
    encoder.encode(hits, families[0], binary_table.astype(np.float32))
    miss_rows = np.flatnonzero(hits == 0)
    if len(miss_rows) == 0:
        return
    classes = np.arange(NUM_CLASSES, dtype=np.int32)
    remaining = np.stack([classes[classes != top[row]] for row in miss_rows])
    miss_table = np.take_along_axis(table[miss_rows], remaining, axis=1)
    miss_symbols = (remaining == symbols[miss_rows, None]).argmax(axis=1).astype(np.int32)
    encoder.encode(miss_symbols, families[1], miss_table.astype(np.float32))


def hierarchical_decode(decoder, families, table: np.ndarray) -> np.ndarray:
    top = table.argmax(axis=1).astype(np.int32)
    rows = np.arange(len(top))
    hit_probability = table[rows, top]
    binary_table = np.stack([1.0 - hit_probability, hit_probability], axis=1)
    hits = decoder.decode(families[0], binary_table.astype(np.float32))
    symbols = top.copy()
    miss_rows = np.flatnonzero(hits == 0)
    if len(miss_rows) == 0:
        return symbols
    classes = np.arange(NUM_CLASSES, dtype=np.int32)
    remaining = np.stack([classes[classes != top[row]] for row in miss_rows])
    miss_table = np.take_along_axis(table[miss_rows], remaining, axis=1)
    miss_symbols = decoder.decode(families[1], miss_table.astype(np.float32))
    symbols[miss_rows] = remaining[np.arange(len(miss_rows)), miss_symbols]
    return symbols


def load_model(args, device):
    sys.path.insert(0, str(args.challenge_root.resolve()))
    sys.path.insert(0, str(args.hpac_source.resolve()))
    from hpac import HPACMini

    packed = deserialize_packed(args.model_blob.read_bytes())
    model = HPACMini(
        num_pairs=N, num_classes=NUM_CLASSES, P=args.patch, delta=args.delta,
        d_film=args.film_dim, ch=args.channels, use_spm=args.spm, b_init=8.0,
    ).eval().to(device)
    missing, unexpected = model.load_state_dict(reconstruct_state(packed), strict=False)
    if unexpected or any(not (key.endswith(".b") or key.endswith(".e")) for key in missing):
        raise ValueError(f"packed model mismatch: missing={missing} unexpected={unexpected}")
    model.set_scn(False)
    return model


def prepare_frame_context(model, idx: torch.Tensor, previous_raw: torch.Tensor):
    """Compute HPAC terms that are invariant across a frame's scan groups."""
    batch, height, width = previous_raw.shape
    patch = model.P
    patch_count = (height // patch) * (width // patch)
    film = model.film_gen(model.frame_embed(idx))
    scale, shift = film.chunk(2, dim=1)
    scale = scale.view(batch, 1, model.ch, 1, 1).expand(
        batch, patch_count, model.ch, 1, 1
    ).reshape(batch * patch_count, model.ch, 1, 1)
    shift = shift.view(batch, 1, model.ch, 1, 1).expand(
        batch, patch_count, model.ch, 1, 1
    ).reshape(batch * patch_count, model.ch, 1, 1)
    previous_one_hot = F.one_hot(
        previous_raw, num_classes=model.num_classes
    ).permute(0, 3, 1, 2).float()
    past_full = model.conv_past(previous_one_hot)
    past = model._to_patches(past_full)
    spm = model._to_patches(model.spm(past_full)) if model.spm is not None else None
    return scale, shift, past, spm


def cached_context_logits(model, current: torch.Tensor, context):
    """Run only the current-residual branch using a prepared frame context."""
    batch, height, width = current.shape
    patch = model.P
    patch_rows, patch_cols = height // patch, width // patch
    patch_count = patch_rows * patch_cols
    one_hot = F.one_hot(
        current, num_classes=model.num_classes
    ).permute(0, 3, 1, 2).float()
    current_patches = model._to_patches(one_hot)
    coords = model._patch_coord_grid(batch * patch_count, current.device)
    hidden = model.gn_a(model.conv_a(torch.cat([current_patches, coords], dim=1)))
    scale, shift, past, spm = context
    hidden = F.gelu(hidden * (1.0 + scale) + shift)
    hidden = hidden + past
    if spm is not None:
        hidden = hidden + spm
    hidden = F.gelu(model.gn_b1(model.conv_b1(hidden)))
    hidden = F.gelu(model.gn_b2(model.conv_b2(hidden)))
    logits = model.head(hidden)
    return model._from_patches(logits, batch, patch_rows, patch_cols)


@torch.no_grad()
def encode(model, raw_tokens, masks, args, device):
    target_symbols = (
        raw_tokens if args.target_mode == "raw" else residuals(raw_tokens)
    )
    encoder = constriction.stream.queue.RangeEncoder()
    family = constriction.stream.model.Categorical(perfect=args.coder_perfect)
    hierarchical_families = (
        constriction.stream.model.Categorical(perfect=args.coder_perfect),
        constriction.stream.model.Categorical(perfect=args.coder_perfect),
    )
    digest = hashlib.sha256()
    previous_raw = torch.zeros((1, H, W), dtype=torch.long, device=device)
    rollout_bits = 0.0
    started = time.time()
    frame_count = raw_tokens.shape[0]
    for frame in range(frame_count):
        idx = torch.tensor([frame], dtype=torch.long, device=device)
        current = torch.zeros((1, H, W), dtype=torch.long, device=device)
        target = target_symbols[frame:frame + 1]
        context = prepare_frame_context(model, idx, previous_raw)
        for mask in masks:
            logits = cached_context_logits(model, current, context)
            selected = logits[0][:, mask].permute(1, 0).contiguous()
            table = probability_table(
                selected, args.logit_precision, digest, args.logit_reference,
                args.round_guard,
            )
            symbols = target[0, mask].cpu().numpy().astype(np.int32)
            rollout_bits += float(
                -np.log2(table[np.arange(len(symbols)), symbols].astype(np.float64)).sum()
            )
            if args.hierarchical:
                hierarchical_encode(encoder, hierarchical_families, table, symbols)
            else:
                encoder.encode(symbols, family, table)
            current[0, mask] = target[0, mask]
        previous_raw = (
            current.clone()
            if args.target_mode == "raw"
            else (current + previous_raw) % NUM_CLASSES
        )
        if frame == 0 or (frame + 1) % 25 == 0:
            print(json.dumps({
                "encoded_frames": frame + 1,
                "elapsed_seconds": time.time() - started,
            }), flush=True)
    return encoder.get_compressed().tobytes(), digest.hexdigest(), rollout_bits


@torch.no_grad()
def decode(model, blob, masks, args, device, frame_count: int):
    decoder = constriction.stream.queue.RangeDecoder(np.frombuffer(blob, dtype=np.uint32))
    family = constriction.stream.model.Categorical(perfect=args.coder_perfect)
    hierarchical_families = (
        constriction.stream.model.Categorical(perfect=args.coder_perfect),
        constriction.stream.model.Categorical(perfect=args.coder_perfect),
    )
    digest = hashlib.sha256()
    raw = torch.empty((frame_count, H, W), dtype=torch.uint8, device="cpu")
    previous_raw = torch.zeros((1, H, W), dtype=torch.long, device=device)
    started = time.time()
    for frame in range(frame_count):
        idx = torch.tensor([frame], dtype=torch.long, device=device)
        current = torch.zeros((1, H, W), dtype=torch.long, device=device)
        context = prepare_frame_context(model, idx, previous_raw)
        for mask in masks:
            logits = cached_context_logits(model, current, context)
            selected = logits[0][:, mask].permute(1, 0).contiguous()
            table = probability_table(
                selected, args.logit_precision, digest, args.logit_reference,
                args.round_guard,
            )
            symbols = (
                hierarchical_decode(decoder, hierarchical_families, table)
                if args.hierarchical
                else decoder.decode(family, table)
            )
            current[0, mask] = torch.from_numpy(symbols.astype(np.int64)).to(device)
        previous_raw = (
            current.clone()
            if args.target_mode == "raw"
            else (current + previous_raw) % NUM_CLASSES
        )
        raw[frame].copy_(previous_raw[0].to(torch.uint8).cpu())
        if frame == 0 or (frame + 1) % 25 == 0:
            print(json.dumps({
                "decoded_frames": frame + 1,
                "elapsed_seconds": time.time() - started,
            }), flush=True)
    return raw, digest.hexdigest()


@torch.no_grad()
def audit_quantizers(model, raw_tokens, masks, args, device):
    specs = (
        ("none_p8", "none", 8, 0.0),
        ("none_p8_g1e-5", "none", 8, 1e-5),
        ("none_p8_g3e-5", "none", 8, 3e-5),
        ("none_p8_g1e-4", "none", 8, 1e-4),
        ("none_p8_g3e-4", "none", 8, 3e-4),
        ("none_p8_g1e-3", "none", 8, 1e-3),
        ("none_p8_g3e-3", "none", 8, 3e-3),
        ("none_p8_g1e-2", "none", 8, 1e-2),
    )
    global_digests = {name: hashlib.sha256() for name, _, _, _ in specs}
    frame_hashes = {name: [] for name, _, _, _ in specs}
    previous_raw = torch.zeros((1, H, W), dtype=torch.long, device=device)
    target_symbols = (
        raw_tokens if args.target_mode == "raw" else residuals(raw_tokens)
    )
    started = time.time()
    for frame in range(raw_tokens.shape[0]):
        idx = torch.tensor([frame], dtype=torch.long, device=device)
        current = torch.zeros((1, H, W), dtype=torch.long, device=device)
        target = target_symbols[frame:frame + 1]
        context = prepare_frame_context(model, idx, previous_raw)
        per_frame = {name: hashlib.sha256() for name, _, _, _ in specs}
        for mask in masks:
            logits = cached_context_logits(model, current, context)
            selected = logits[0][:, mask].permute(1, 0).contiguous()
            for name, reference, precision, guard in specs:
                raw = quantize_logits(
                    selected, precision, reference, guard
                ).tobytes(order="C")
                global_digests[name].update(raw)
                per_frame[name].update(raw)
            current[0, mask] = target[0, mask]
        previous_raw = (
            current.clone()
            if args.target_mode == "raw"
            else (current + previous_raw) % NUM_CLASSES
        )
        for name, _, _, _ in specs:
            frame_hashes[name].append(per_frame[name].hexdigest())
        if frame == 0 or (frame + 1) % 25 == 0:
            print(json.dumps({
                "audited_frames": frame + 1,
                "elapsed_seconds": time.time() - started,
            }), flush=True)
    return {
        "frames": raw_tokens.shape[0],
        "global_hashes": {
            name: global_digests[name].hexdigest() for name, _, _, _ in specs
        },
        "frame_hashes": frame_hashes,
    }


@torch.no_grad()
def greedy_rollout(model, expected_raw, masks, device):
    generated = torch.empty_like(expected_raw, dtype=torch.uint8, device="cpu")
    previous_raw = torch.zeros((1, H, W), dtype=torch.long, device=device)
    mismatch_count = 0
    quantized_top_mismatch_count = 0
    diagnostic_bits = 0.0
    frame_mismatch = []
    started = time.time()
    for frame in range(expected_raw.shape[0]):
        idx = torch.tensor([frame], dtype=torch.long, device=device)
        current = torch.zeros((1, H, W), dtype=torch.long, device=device)
        context = prepare_frame_context(model, idx, previous_raw)
        for mask in masks:
            logits = cached_context_logits(model, current, context)
            selected = logits[0][:, mask].permute(1, 0).contiguous()
            expected_symbols = expected_raw[frame, mask].cpu().numpy().astype(np.int32)
            quantized = quantize_logits(selected, 8, "max")
            quantized_logits = quantized.astype(np.float64) / 8
            quantized_logits -= quantized_logits.max(axis=1, keepdims=True)
            table = np.exp(quantized_logits)
            table /= table.sum(axis=1, keepdims=True)
            quantized_top = table.argmax(axis=1).astype(np.int64)
            quantized_top_mismatch_count += int(
                (quantized_top != expected_symbols).sum()
            )
            diagnostic_bits += float(-np.log2(
                table[np.arange(len(expected_symbols)), expected_symbols]
            ).sum())
            current[0, mask] = torch.from_numpy(quantized_top).to(device)
        expected = expected_raw[frame]
        mismatches = int((current[0] != expected).sum().item())
        mismatch_count += mismatches
        frame_mismatch.append(mismatches)
        previous_raw = current
        generated[frame].copy_(current[0].to(torch.uint8).cpu())
        if frame == 0 or (frame + 1) % 25 == 0:
            print(json.dumps({
                "greedy_frames": frame + 1,
                "mismatches": mismatch_count,
                "elapsed_seconds": time.time() - started,
            }), flush=True)
    return generated, {
        "frames": expected_raw.shape[0],
        "mismatch_count": mismatch_count,
        "mismatch_rate": mismatch_count / expected_raw.numel(),
        "quantized_top_mismatch_count": quantized_top_mismatch_count,
        "diagnostic_bits": diagnostic_bits,
        "frame_mismatch": frame_mismatch,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge-root", type=Path, required=True)
    parser.add_argument("--hpac-source", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--model-blob", type=Path, required=True)
    parser.add_argument("--patch", type=int, default=32)
    parser.add_argument("--delta", type=int, default=2)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--film-dim", type=int, default=8)
    parser.add_argument("--spm", action="store_true")
    parser.add_argument("--logit-precision", type=int, default=8)
    parser.add_argument(
        "--round-guard", type=float, default=0.0,
        help="force logits near half-integer rounding boundaries to the lower code",
    )
    parser.add_argument(
        "--logit-reference", choices=("none", "class0", "max"), default="none",
        help="subtract a softmax-invariant reference before logit quantization",
    )
    parser.add_argument(
        "--target-mode", choices=("raw", "residual"), default="residual"
    )
    parser.add_argument("--coder-perfect", action="store_true")
    parser.add_argument("--hierarchical", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--conv-backend", choices=("cudnn", "aten"), default="cudnn",
        help="use the common ATen convolution path instead of architecture-specific cuDNN",
    )
    parser.add_argument(
        "--frames", type=int, default=N,
        help="encode only the first N frames for bounded portability benchmarks",
    )
    parser.add_argument(
        "--threads", type=int, default=1,
        help="PyTorch CPU thread count; one is the deterministic portability default",
    )
    parser.add_argument("--tokens-out", type=Path)
    parser.add_argument(
        "--decode-from", type=Path,
        help="decode an existing stream instead of encoding from --cache",
    )
    parser.add_argument(
        "--audit-quantizers", action="store_true",
        help="teacher-force identical tokens and hash several quantizers in one pass",
    )
    parser.add_argument(
        "--greedy-rollout", action="store_true",
        help="generate raw tokens by HPAC argmax with no entropy stream",
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    if not 1 <= args.frames <= N:
        parser.error(f"--frames must be between 1 and {N}")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.conv_backend == "aten":
        torch.backends.cudnn.enabled = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    model = load_model(args, device)
    masks = group_masks(args.patch, args.delta, device)
    if args.greedy_rollout:
        if args.cache is None:
            parser.error("--greedy-rollout requires --cache")
        expected = torch.load(
            args.cache, map_location="cpu", weights_only=False
        )["seg"][:args.frames].long().to(device)
        generated, result = greedy_rollout(model, expected, masks, device)
        raw_bytes = generated.numpy().tobytes(order="C")
        result["raw_token_sha256"] = hashlib.sha256(raw_bytes).hexdigest()
        if args.tokens_out is not None:
            args.tokens_out.parent.mkdir(parents=True, exist_ok=True)
            args.tokens_out.write_bytes(raw_bytes)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({
            key: value for key, value in result.items() if key != "frame_mismatch"
        }, indent=2), flush=True)
        return
    if args.audit_quantizers:
        if args.cache is None:
            parser.error("--audit-quantizers requires --cache")
        raw = torch.load(args.cache, map_location="cpu", weights_only=False)["seg"]
        raw = raw[:args.frames].long().to(device)
        result = audit_quantizers(model, raw, masks, args, device)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result["global_hashes"], indent=2), flush=True)
        return
    if args.decode_from is not None:
        decoded, decode_hash = decode(
            model, args.decode_from.read_bytes(), masks, args, device, args.frames
        )
        raw_bytes = decoded.numpy().tobytes(order="C")
        result = {
            "decoded_frames": args.frames,
            "logit_hash_decode": decode_hash,
            "raw_token_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        }
        if args.cache is not None:
            expected = torch.load(
                args.cache, map_location="cpu", weights_only=False
            )["seg"][:args.frames].to(torch.uint8)
            result["verified_exact"] = bool(torch.equal(decoded, expected))
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        return
    if args.cache is None or args.tokens_out is None:
        parser.error("encoding requires --cache and --tokens-out")
    raw = torch.load(args.cache, map_location="cpu", weights_only=False)["seg"]
    raw = raw[:args.frames].long().to(device)
    blob, encode_hash, rollout_bits = encode(model, raw, masks, args, device)
    args.tokens_out.parent.mkdir(parents=True, exist_ok=True)
    args.tokens_out.write_bytes(blob)
    result = {
        "token_bytes": len(blob),
        "token_bpp": len(blob) * 8 / raw.numel(),
        "rollout_bpp": rollout_bits / raw.numel(),
        "logit_hash_encode": encode_hash,
    }
    if args.verify:
        decoded, decode_hash = decode(model, blob, masks, args, device, args.frames)
        expected = raw.to(torch.uint8).cpu()
        result["logit_hash_decode"] = decode_hash
        result["verified_exact"] = bool(torch.equal(decoded, expected))
        if not result["verified_exact"] or decode_hash != encode_hash:
            raise RuntimeError("HPAC round trip or deterministic-logit verification failed")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
