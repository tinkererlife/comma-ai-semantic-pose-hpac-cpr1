#!/usr/bin/env python3
"""Pack IntegerHPAC parameters into the exact deployment representation."""

from __future__ import annotations

import argparse
import json
import lzma
import math
import struct
from pathlib import Path

import numpy as np
import torch

from hpac_integer import IntegerConv2d, IntegerHPAC


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
BITPACK_MAGIC = b"IHB1"


def packed_parameters(model: IntegerHPAC):
    modules = dict(model.named_modules())
    for name, parameter in model.named_parameters():
        module_name, field = name.rsplit(".", 1)
        module = modules[module_name]
        if field == "bias":
            low, high = -32768, 32767
        elif field == "exponent":
            low, high = module.exponent_min, 0
        elif field == "gate":
            low, high = 0, module.gate_bound
        else:
            low = -getattr(module, "weight_bound", 127)
            high = getattr(module, "weight_bound", 127)
        value = parameter.detach().round().clamp(low, high)
        if field == "weight" and isinstance(module, IntegerConv2d):
            mask = module.mask.to(torch.bool).expand_as(value)
            value = value[mask]
        dtype = np.dtype("<i2" if field == "bias" else "i1")
        yield name, tuple(parameter.shape), dtype, value.cpu().numpy().astype(dtype)


def serialize(model: IntegerHPAC) -> bytes:
    return b"".join(value.tobytes(order="C") for _, _, _, value in packed_parameters(model))


def serialize_bitpacked(model: IntegerHPAC) -> bytes:
    chunks = [BITPACK_MAGIC]
    for _, _, _, value in packed_parameters(model):
        flat = value.reshape(-1).astype(np.int64)
        low = int(flat.min())
        span = int(flat.max()) - low
        bits = int(math.ceil(math.log2(span + 1))) if span else 0
        chunks.append(struct.pack("<hB", low, bits))
        if bits:
            codes = (flat - low).astype(np.uint64)
            bit_rows = (
                (codes[:, None] >> np.arange(bits, dtype=np.uint64)) & 1
            ).astype(np.uint8)
            chunks.append(np.packbits(
                bit_rows.reshape(-1), bitorder="little"
            ).tobytes())
    return b"".join(chunks)


def deserialize_bitpacked(model: IntegerHPAC, raw: bytes) -> None:
    modules = dict(model.named_modules())
    offset = len(BITPACK_MAGIC)
    with torch.no_grad():
        for name, shape, _, packed in packed_parameters(model):
            low, bits = struct.unpack_from("<hB", raw, offset)
            offset += 3
            count = packed.size
            byte_count = (count * bits + 7) // 8
            if bits:
                payload = np.frombuffer(
                    raw, dtype=np.uint8, count=byte_count, offset=offset
                )
                unpacked = np.unpackbits(payload, bitorder="little")[:count * bits]
                bit_rows = unpacked.reshape(count, bits).astype(np.uint64)
                codes = (
                    bit_rows * (1 << np.arange(bits, dtype=np.uint64))
                ).sum(axis=1, dtype=np.uint64).astype(np.int64)
                value = codes + low
            else:
                value = np.full(count, low, dtype=np.int64)
            _restore_parameter(model, modules, name, shape, value)
            offset += byte_count
    if offset != len(raw):
        raise ValueError(f"bitpacked model has {len(raw) - offset} trailing bytes")


def _restore_parameter(model, modules, name, shape, value) -> None:
    module_name, field = name.rsplit(".", 1)
    module = modules[module_name]
    parameter = module._parameters[field]
    if field == "weight" and isinstance(module, IntegerConv2d):
        restored = torch.zeros(shape, dtype=parameter.dtype)
        mask = module.mask.to(torch.bool).expand(*shape)
        restored[mask] = torch.from_numpy(value.astype(np.float32))
    else:
        restored = torch.from_numpy(value.reshape(shape).astype(np.float32))
    parameter.copy_(restored)


def deserialize(model: IntegerHPAC, raw: bytes) -> None:
    if raw.startswith(BITPACK_MAGIC):
        deserialize_bitpacked(model, raw)
        return
    modules = dict(model.named_modules())
    offset = 0
    with torch.no_grad():
        for name, shape, dtype, packed in packed_parameters(model):
            count = packed.size
            size = count * dtype.itemsize
            value = np.frombuffer(raw, dtype=dtype, count=count, offset=offset).copy()
            _restore_parameter(model, modules, name, shape, value)
            offset += size
    if offset != len(raw):
        raise ValueError(f"packed model has {len(raw) - offset} trailing bytes")


def load_model(path: Path, args) -> IntegerHPAC:
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
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
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
    parser.add_argument("--norm-gates", action="store_true")
    parser.add_argument("--blob", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    model = load_model(args.checkpoint, args)
    byte_raw = serialize(model)
    bit_raw = serialize_bitpacked(model)
    candidates = {
        "byte": (
            byte_raw,
            lzma.compress(byte_raw, format=lzma.FORMAT_XZ, filters=LZMA_FILTERS),
        ),
        "bitpack": (
            bit_raw,
            lzma.compress(bit_raw, format=lzma.FORMAT_XZ, filters=LZMA_FILTERS),
        ),
    }
    layout, (raw, blob) = min(
        candidates.items(), key=lambda item: len(item[1][1])
    )

    restored = IntegerHPAC(
        channels=args.channels, patch=args.patch,
        delta=args.delta, frame_dim=args.frame_dim,
        norm_mode=args.norm_mode, activation=args.activation,
        use_frame_scale=args.frame_scale,
        weight_bound=args.weight_bound, activation_bound=args.activation_bound,
        use_weight_scales=args.weight_scales,
        weight_exponent_min=args.weight_exponent_min,
        use_spm=args.spm,
        use_norm_gates=args.norm_gates,
    ).eval()
    deserialize(restored, lzma.decompress(blob))
    expected = serialize(model)
    actual = serialize(restored)
    if actual != expected:
        raise RuntimeError("integer model changed during pack/unpack round trip")

    result = {
        "raw_model_bytes": len(raw),
        "compressed_model_bytes": len(blob),
        "layout": layout,
        "candidate_bytes": {
            name: {"raw": len(candidate_raw), "xz": len(candidate_blob)}
            for name, (candidate_raw, candidate_blob) in candidates.items()
        },
        "verified_exact": True,
    }
    args.blob.parent.mkdir(parents=True, exist_ok=True)
    args.blob.write_bytes(blob)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
