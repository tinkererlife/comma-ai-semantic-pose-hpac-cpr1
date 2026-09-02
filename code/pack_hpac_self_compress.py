#!/usr/bin/env python3
"""Pack learned per-output-channel HPAC bit depths exactly."""

from __future__ import annotations

import argparse
import json
import lzma
import math
from pathlib import Path

import numpy as np
import torch

from hpac_integer import IntegerConv2d, IntegerHPAC, IntegerLinear
from hpac_self_compress import enable_self_compression, set_deployed_bit_depths


MAGIC = b"IHS1"
COMPRESSIBLE_TYPES = (IntegerConv2d, IntegerLinear)
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


def compressible_modules(model):
    return [
        module for module in model.modules()
        if isinstance(module, COMPRESSIBLE_TYPES)
    ]


def deployed_depths(module) -> np.ndarray:
    return (
        module.bit_depth.detach().relu()
        .clamp(max=module.self_compress_max_bits)
        .round().to(torch.uint8).cpu().numpy()
    )


def pack_nibbles(values: np.ndarray) -> bytes:
    values = values.astype(np.uint8, copy=False).reshape(-1)
    if np.any(values > 15):
        raise ValueError("nibble metadata exceeds 4 bits")
    output = np.zeros((len(values) + 1) // 2, dtype=np.uint8)
    output[: len(values[0::2])] |= values[0::2]
    output[: len(values[1::2])] |= values[1::2] << 4
    return output.tobytes()


def unpack_nibbles(raw: memoryview, count: int) -> tuple[np.ndarray, memoryview]:
    byte_count = (count + 1) // 2
    packed = np.frombuffer(raw[:byte_count], dtype=np.uint8)
    values = np.empty(byte_count * 2, dtype=np.uint8)
    values[0::2] = packed & 0xF
    values[1::2] = packed >> 4
    return values[:count].copy(), raw[byte_count:]


def module_weight_rows(module, weight: torch.Tensor) -> list[np.ndarray]:
    if isinstance(module, IntegerConv2d):
        mask = module.mask.to(torch.bool).expand_as(weight)
        return [
            weight[index][mask[index]].detach().cpu().numpy().astype(np.int16)
            for index in range(weight.shape[0])
        ]
    return [
        weight[index].reshape(-1).detach().cpu().numpy().astype(np.int16)
        for index in range(weight.shape[0])
    ]


def serialize_self_compressed(model) -> bytes:
    modules = compressible_modules(model)
    depths = [deployed_depths(module) for module in modules]
    weight_bits: list[np.ndarray] = []
    for module, module_depths in zip(modules, depths):
        weight = module.codes()[0]
        for bits, row in zip(module_depths.tolist(), module_weight_rows(module, weight)):
            if bits == 0:
                if np.any(row != 0):
                    raise ValueError("zero-bit channel contains nonzero weights")
                continue
            low, high = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
            if int(row.min(initial=0)) < low or int(row.max(initial=0)) > high:
                raise ValueError(f"{bits}-bit channel exceeds signed range")
            codes = row.astype(np.int64) & ((1 << bits) - 1)
            weight_bits.append(
                ((codes[:, None] >> np.arange(bits, dtype=np.int64)) & 1)
                .astype(np.uint8).reshape(-1)
            )
    packed_weights = (
        np.packbits(np.concatenate(weight_bits), bitorder="little").tobytes()
        if weight_bits else b""
    )

    module_by_name = dict(model.named_modules())
    fixed = []
    for name, parameter in model.named_parameters():
        module_name, field = name.rsplit(".", 1)
        module = module_by_name[module_name]
        if field == "bit_depth" or (
            field == "weight" and isinstance(module, COMPRESSIBLE_TYPES)
        ):
            continue
        if field == "bias":
            value = parameter.detach().round().clamp(-32768, 32767)
            dtype = np.dtype("<i2")
        elif field == "exponent":
            value = parameter.detach().round().clamp(module.exponent_min, 0)
            dtype = np.dtype("i1")
        else:
            bound = getattr(module, "weight_bound", 127)
            value = parameter.detach().round().clamp(-bound, bound)
            dtype = np.dtype("i1")
        fixed.append(value.cpu().numpy().astype(dtype).tobytes(order="C"))

    metadata = pack_nibbles(np.concatenate(depths))
    return MAGIC + metadata + packed_weights + b"".join(fixed)


def restore_weight_row(module, parameter, index: int, values: np.ndarray) -> None:
    if isinstance(module, IntegerConv2d):
        mask = module.mask.to(torch.bool).expand_as(parameter)[index]
        parameter[index].zero_()
        parameter[index][mask] = torch.from_numpy(values.astype(np.float32))
    else:
        parameter[index].copy_(
            torch.from_numpy(values.reshape(parameter[index].shape).astype(np.float32))
        )


def deserialize_self_compressed(model: IntegerHPAC, raw: bytes) -> None:
    if not raw.startswith(MAGIC):
        raise ValueError("not a self-compressed HPAC model")
    view = memoryview(raw)[len(MAGIC):]
    modules = compressible_modules(model)
    channel_count = sum(module.weight.shape[0] for module in modules)
    all_depths, view = unpack_nibbles(view, channel_count)
    depth_offset = 0
    total_weight_bits = 0
    for module in modules:
        module_depths = all_depths[depth_offset:depth_offset + module.weight.shape[0]]
        row_counts = [len(row) for row in module_weight_rows(module, module.weight)]
        total_weight_bits += sum(
            int(bits) * count for bits, count in zip(module_depths, row_counts)
        )
        depth_offset += module.weight.shape[0]
    weight_bytes = (total_weight_bits + 7) // 8
    packed = np.frombuffer(view[:weight_bytes], dtype=np.uint8)
    bits_view = np.unpackbits(packed, bitorder="little")[:total_weight_bits]
    view = view[weight_bytes:]

    bit_offset = 0
    depth_offset = 0
    with torch.no_grad():
        for module in modules:
            parameter = module.weight
            module_depths = all_depths[
                depth_offset:depth_offset + parameter.shape[0]
            ]
            rows = module_weight_rows(module, parameter)
            for index, (bits, template) in enumerate(zip(module_depths, rows)):
                count = len(template)
                bits = int(bits)
                if bits:
                    count_bits = count * bits
                    bit_rows = bits_view[bit_offset:bit_offset + count_bits].reshape(
                        count, bits
                    ).astype(np.int16)
                    unsigned = (
                        bit_rows * (1 << np.arange(bits, dtype=np.int16))
                    ).sum(axis=1, dtype=np.int16)
                    sign = 1 << (bits - 1)
                    values = np.where(
                        unsigned >= sign, unsigned - (1 << bits), unsigned
                    ).astype(np.int16)
                    bit_offset += count_bits
                else:
                    values = np.zeros(count, dtype=np.int16)
                restore_weight_row(module, parameter, index, values)
            depth_offset += parameter.shape[0]

        module_by_name = dict(model.named_modules())
        for name, parameter in model.named_parameters():
            module_name, field = name.rsplit(".", 1)
            module = module_by_name[module_name]
            if field == "weight" and isinstance(module, COMPRESSIBLE_TYPES):
                continue
            dtype = np.dtype("<i2" if field == "bias" else "i1")
            byte_count = parameter.numel() * dtype.itemsize
            value = np.frombuffer(
                view[:byte_count], dtype=dtype, count=parameter.numel()
            ).copy().reshape(parameter.shape)
            parameter.copy_(torch.from_numpy(value.astype(np.float32)))
            view = view[byte_count:]
    if bit_offset != total_weight_bits or len(view):
        raise ValueError("self-compressed HPAC payload has trailing data")


def model_from_args(args, self_compressed: bool):
    model = IntegerHPAC(
        channels=args.channels,
        patch=args.patch,
        delta=args.delta,
        frame_dim=args.frame_dim,
        norm_mode="none",
        activation="relu",
        use_frame_scale=True,
        weight_bound=args.weight_bound,
        activation_bound=args.activation_bound,
        use_weight_scales=True,
        weight_exponent_min=args.weight_exponent_min,
        use_spm=True,
    )
    if self_compressed:
        enable_self_compression(model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--patch", type=int, default=64)
    parser.add_argument("--delta", type=int, default=2)
    parser.add_argument("--frame-dim", type=int, default=8)
    parser.add_argument("--weight-bound", type=int, default=127)
    parser.add_argument("--activation-bound", type=int, default=127)
    parser.add_argument("--weight-exponent-min", type=int, default=-6)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--blob", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    source = model_from_args(args, True)
    source.load_state_dict(checkpoint["state_dict"])
    set_deployed_bit_depths(source, True)
    source.eval()
    raw = serialize_self_compressed(source)
    restored = model_from_args(args, False).eval()
    deserialize_self_compressed(restored, raw)

    device = torch.device(args.device)
    source = source.to(device)
    restored = restored.to(device)
    generator = torch.Generator(device=device).manual_seed(20260716)
    current = torch.randint(
        0, 5, (2, 384, 512), generator=generator, device=device
    )
    previous = torch.randint(
        0, 5, (2, 384, 512), generator=generator, device=device
    )
    idx = torch.tensor([0, 599], device=device)
    with torch.no_grad():
        expected = source(current, idx, previous)
        actual = restored(current, idx, previous)
    max_diff = float((expected - actual).abs().max())
    if max_diff != 0.0:
        mismatches = {}
        restored_modules = dict(restored.named_modules())
        for name, module in source.named_modules():
            if not isinstance(module, COMPRESSIBLE_TYPES):
                continue
            other = restored_modules[name]
            fields = []
            for expected_field, actual_field in zip(
                module.codes(), other.codes(), strict=True
            ):
                if expected_field is None:
                    continue
                expected_cpu = expected_field.detach().cpu()
                actual_cpu = actual_field.detach().cpu()
                delta = expected_cpu - actual_cpu
                indices = (delta != 0).nonzero()[:8]
                fields.append({
                    "max": float(delta.abs().max()),
                    "count": int((delta != 0).sum()),
                    "indices": indices.tolist(),
                    "expected": [float(expected_cpu[tuple(index)]) for index in indices],
                    "actual": [float(actual_cpu[tuple(index)]) for index in indices],
                })
            if any(field["count"] for field in fields):
                mismatches[name] = {
                    "fields": fields,
                    "depths": deployed_depths(module).tolist(),
                }
        frame_diff = float(
            (source.frame_codes().detach().cpu()
             - restored.frame_codes().detach().cpu()).abs().max()
        )
        if frame_diff:
            mismatches["frame_embed"] = {"max": frame_diff}
        raise RuntimeError(
            "self-compressed round trip changed logits by "
            f"{max_diff}; effective-code mismatches={mismatches}"
        )

    blob = lzma.compress(raw, format=lzma.FORMAT_XZ, filters=LZMA_FILTERS)
    result = {
        "raw_model_bytes": len(raw),
        "compressed_model_bytes": len(blob),
        "metadata_bytes": math.ceil(sum(
            module.weight.shape[0] for module in compressible_modules(source)
        ) / 2),
        "verified_exact": True,
        "max_logit_diff": max_diff,
    }
    args.blob.parent.mkdir(parents=True, exist_ok=True)
    args.blob.write_bytes(blob)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
