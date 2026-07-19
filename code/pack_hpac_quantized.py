#!/usr/bin/env python3
"""Pack a trained HPAC checkpoint and measure its quantized entropy rate."""

from __future__ import annotations

import argparse
import json
import lzma
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


N = 600
NUM_CLASSES = 5
MASKED_SPECS = {
    "conv_a": ("A", 2, 7),
    "conv_b1": ("B", 2, 5),
    "conv_b2": ("B", 2, 3),
}
PACKED_SCHEMA = (
    ("conv_a.weight_q", (64, 7, 23), "i1"),
    ("conv_a.weight_scale", (64,), "<f2"),
    ("conv_b1.weight_q", (64, 1, 14), "i1"),
    ("conv_b1.weight_scale", (64,), "<f2"),
    ("conv_b2.weight_q", (64, 1, 5), "i1"),
    ("conv_b2.weight_scale", (64,), "<f2"),
    ("conv_past.weight_q", (64, 5, 3, 3), "i1"),
    ("conv_past.weight_scale", (64,), "<f2"),
    ("film_gen.weight_q", (128, 8), "i1"),
    ("film_gen.weight_scale", (128,), "<f2"),
    ("head.weight_q", (5, 64, 1, 1), "i1"),
    ("head.weight_scale", (5,), "<f2"),
    ("spm.dw.weight_q", (64, 1, 3, 3), "i1"),
    ("spm.dw.weight_scale", (64,), "<f2"),
    ("spm.pw.weight_q", (64, 64, 1, 1), "i1"),
    ("spm.pw.weight_scale", (64,), "<f2"),
    ("frame_embed.weight_q", (600, 8), "i1"),
    ("frame_embed.weight_scale", (8,), "<f2"),
    ("film_gen.bias", (128,), "<f2"),
    ("conv_a.bias", (64,), "<f2"),
    ("gn_a.scale", (64,), "<f2"),
    ("gn_a.shift", (64,), "<f2"),
    ("conv_b1.bias", (64,), "<f2"),
    ("gn_b1.scale", (64,), "<f2"),
    ("gn_b1.shift", (64,), "<f2"),
    ("conv_b2.bias", (64,), "<f2"),
    ("gn_b2.scale", (64,), "<f2"),
    ("gn_b2.shift", (64,), "<f2"),
    ("conv_past.bias", (64,), "<f2"),
    ("spm.norm.scale", (64,), "<f2"),
    ("spm.norm.shift", (64,), "<f2"),
    ("spm.dw.bias", (64,), "<f2"),
    ("spm.pw.bias", (64,), "<f2"),
    ("head.bias", (5,), "<f2"),
)
LZMA_FILTERS = [{
    "id": lzma.FILTER_LZMA2,
    "dict_size": 1 << 16,
    "lc": 0,
    "lp": 1,
    "pb": 0,
    "mode": lzma.MODE_NORMAL,
    "nice_len": 273,
    "mf": lzma.MF_BT4,
    "depth": 0,
}]


def serialize_packed(packed: dict[str, torch.Tensor]) -> bytes:
    if set(packed) != {name for name, _, _ in PACKED_SCHEMA}:
        raise ValueError("packed HPAC state does not match fixed deployment schema")
    chunks = []
    for name, shape, dtype in PACKED_SCHEMA:
        value = packed[name].detach().cpu()
        if tuple(value.shape) != shape:
            raise ValueError(f"unexpected packed shape for {name}: {tuple(value.shape)}")
        chunks.append(value.numpy().astype(dtype, copy=False).tobytes())
    return b"".join(chunks)


def deserialize_packed(blob: bytes) -> dict[str, torch.Tensor]:
    raw = lzma.decompress(blob)
    packed = {}
    offset = 0
    for name, shape, dtype in PACKED_SCHEMA:
        np_dtype = np.dtype(dtype)
        count = math.prod(shape)
        byte_count = count * np_dtype.itemsize
        array = np.frombuffer(raw, dtype=np_dtype, count=count, offset=offset).copy()
        packed[name] = torch.from_numpy(array.reshape(shape))
        offset += byte_count
    if offset != len(raw):
        raise ValueError("packed HPAC blob has trailing bytes")
    return packed


def deployment_mask(base: str) -> torch.Tensor | None:
    if base not in MASKED_SPECS:
        return None
    type_, delta, kernel = MASKED_SPECS[base]
    center = (kernel - 1) // 2
    mask = torch.zeros(kernel, kernel, dtype=torch.bool)
    for row in range(kernel):
        for col in range(kernel):
            offset = col - center + delta * (row - center)
            if offset < 0 or (type_ == "B" and offset == 0):
                mask[row, col] = True
    return mask.view(1, 1, kernel, kernel)


def residuals(tokens: torch.Tensor) -> torch.Tensor:
    result = torch.empty_like(tokens)
    result[0] = tokens[0]
    result[1:] = (tokens[1:] - tokens[:-1]) % NUM_CLASSES
    return result


def pack_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    packed = {}
    bases = sorted({
        key[:-2] for key in state
        if key.endswith(".b") and key[:-2] + ".e" in state
    })
    skipped = set()
    for base in bases:
        weight_key = base + ".weight"
        if weight_key not in state:
            continue
        weight = state[weight_key].float()
        bits = state[base + ".b"].float().round().clamp(2, 8)
        if not bool((bits == 8).all()):
            raise ValueError(f"{base} is not configured for fixed int8 packing")
        scale = torch.pow(2.0, state[base + ".e"].float()).to(torch.float16)
        shape = [1] * weight.ndim
        shape[0] = -1
        q = (weight / scale.float().view(*shape)).round().clamp(-128, 127).to(torch.int8)
        mask = deployment_mask(base)
        if mask is not None:
            q = q.flatten(2)[:, :, mask.flatten()]
        packed[base + ".weight_q"] = q
        packed[base + ".weight_scale"] = scale
        skipped.update({weight_key, base + ".b", base + ".e"})

    embedding = state["frame_embed.weight"].float()
    embedding_scale = embedding.abs().amax(0).clamp_min(1e-8) / 127.0
    embedding_scale = embedding_scale.to(torch.float16)
    packed["frame_embed.weight_q"] = (
        embedding / embedding_scale.float()[None]
    ).round().clamp(-127, 127).to(torch.int8)
    packed["frame_embed.weight_scale"] = embedding_scale
    skipped.add("frame_embed.weight")

    for key, value in state.items():
        if key in skipped:
            continue
        packed[key] = value.to(torch.float16) if torch.is_floating_point(value) else value
    return packed


def reconstruct_state(packed: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    state = {}
    bases = sorted({
        key[:-len(".weight_q")] for key in packed
        if key.endswith(".weight_q") and key != "frame_embed.weight_q"
    })
    skipped = set()
    for base in bases:
        q = packed[base + ".weight_q"].float()
        mask = deployment_mask(base)
        if mask is not None:
            kernel = mask.shape[-1]
            full = torch.zeros(q.shape[0], q.shape[1], kernel * kernel)
            full[:, :, mask.flatten()] = q
            q = full.reshape(q.shape[0], q.shape[1], kernel, kernel)
        scale = packed[base + ".weight_scale"].float()
        shape = [1] * q.ndim
        shape[0] = -1
        state[base + ".weight"] = q * scale.view(*shape)
        skipped.update({base + ".weight_q", base + ".weight_scale"})
    state["frame_embed.weight"] = (
        packed["frame_embed.weight_q"].float()
        * packed["frame_embed.weight_scale"].float()[None]
    )
    skipped.update({"frame_embed.weight_q", "frame_embed.weight_scale"})
    for key, value in packed.items():
        if key not in skipped:
            state[key] = value.float() if torch.is_floating_point(value) else value
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge-root", type=Path, required=True)
    parser.add_argument("--hpac-source", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--blob", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    sys.path.insert(0, str(args.challenge_root.resolve()))
    sys.path.insert(0, str(args.hpac_source.resolve()))
    from hpac import HPACMini

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["result"]["config"]
    packed = pack_state(checkpoint["state_dict"])
    serialized = serialize_packed(packed)
    blob = lzma.compress(
        serialized, format=lzma.FORMAT_XZ, filters=LZMA_FILTERS
    )
    args.blob.parent.mkdir(parents=True, exist_ok=True)
    args.blob.write_bytes(blob)

    model = HPACMini(
        num_pairs=N, num_classes=NUM_CLASSES,
        P=int(config["patch"]), delta=int(config["delta"]),
        d_film=int(config["film_dim"]), ch=int(config["channels"]),
        use_spm=bool(config["spm"]), b_init=8.0,
    ).eval().to(device)
    missing, unexpected = model.load_state_dict(reconstruct_state(packed), strict=False)
    allowed_missing = {key for key in missing if key.endswith(".b") or key.endswith(".e")}
    if len(allowed_missing) != len(missing) or unexpected:
        raise ValueError(f"state reconstruction mismatch: missing={missing} unexpected={unexpected}")
    model.set_scn(False)

    tokens = torch.load(args.cache, map_location="cpu", weights_only=False)["seg"].long().to(device)
    target_mode = config.get("target_mode", "residual")
    target = tokens if target_mode == "raw" else residuals(tokens)
    previous = torch.zeros_like(tokens)
    previous[1:] = tokens[:-1]
    nats = 0.0
    with torch.inference_mode():
        for start in range(0, N, args.eval_batch_size):
            end = min(start + args.eval_batch_size, N)
            idx = torch.arange(start, end, device=device)
            logits = model(target[start:end], idx, previous[start:end])
            nats += float(F.cross_entropy(logits, target[start:end], reduction="sum"))
    bpp = nats / math.log(2.0) / target.numel()
    result = {
        "target_mode": target_mode,
        "quantized_bpp": bpp,
        "estimated_token_bytes": math.ceil(bpp * target.numel() / 8),
        "custom_serialized_bytes": len(serialized),
        "lzma_model_bytes": len(blob),
        "projected_model_plus_tokens_bytes": len(blob) + math.ceil(bpp * target.numel() / 8),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
