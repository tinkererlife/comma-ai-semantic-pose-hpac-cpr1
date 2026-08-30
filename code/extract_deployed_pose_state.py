#!/usr/bin/env python3
"""Extract exact deployed carrier state and rendered master frames for search."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import zipfile
from pathlib import Path

import numpy as np
import torch

from carrier_codec import decode_compact_carrier
from model_bundle import decode_model_bundle, parse_model_field


FRAMES = 600
DIMENSIONS = 12
BASIS_SHAPE = (DIMENSIONS, 3, 24, 32)
CAMERA_SHAPE = (874, 1164, 3)


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _absolute_codes(encoded: np.ndarray) -> np.ndarray:
    delta = (encoded >> 1) ^ -(encoded & 1)
    unsigned = np.cumsum(delta.astype(np.int64), axis=0) & 0xFFF
    return np.where(unsigned >= 0x800, unsigned - 0x1000, unsigned).astype(
        np.int16
    )


def extract(archive: Path, raw: Path, master_out: Path, carrier_out: Path) -> dict:
    with zipfile.ZipFile(archive) as source:
        payload = source.read("p")
    model_bytes, flags = parse_model_field(struct.unpack_from("<I", payload)[0])
    bundle = decode_model_bundle(payload[4:4 + model_bytes], flags)
    basis_count = int(np.prod(BASIS_SHAPE))
    basis_scales, basis_codes, coeff_scales, encoded = decode_compact_carrier(
        bundle.carrier,
        basis_count=basis_count,
        frames=FRAMES,
        dimensions=DIMENSIONS,
    )
    coeff_codes = _absolute_codes(encoded)
    basis = (
        basis_codes.reshape(BASIS_SHAPE).astype(np.float32)
        * basis_scales[:, None, None, None]
    )
    coeff = coeff_codes.astype(np.float32) * coeff_scales[None]

    expected_raw_bytes = 2 * FRAMES * int(np.prod(CAMERA_SHAPE))
    if raw.stat().st_size != expected_raw_bytes:
        raise ValueError(
            f"inflated raw size mismatch: {raw.stat().st_size} != {expected_raw_bytes}"
        )
    frames = np.memmap(raw, mode="r", dtype=np.uint8, shape=(2 * FRAMES,) + CAMERA_SHAPE)
    masters = torch.from_numpy(np.asarray(frames[1::2]).copy()).permute(0, 3, 1, 2)

    master_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"masters": masters, "source_archive_sha256": _sha256(archive)}, master_out)
    carrier_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "basis": torch.from_numpy(basis),
        "basis_codes": torch.from_numpy(basis_codes.reshape(BASIS_SHAPE).copy()),
        "basis_scales": torch.from_numpy(basis_scales.copy()),
        "coeff": torch.from_numpy(coeff),
        "coeff_codes": torch.from_numpy(coeff_codes),
        "coeff_scales": torch.from_numpy(coeff_scales.copy()),
        "canonical_carrier_sha256": hashlib.sha256(bundle.carrier).hexdigest(),
        "source_archive_sha256": _sha256(archive),
    }, carrier_out)
    return {
        "archive_sha256": _sha256(archive),
        "raw_sha256": _sha256(raw),
        "master_cache": str(master_out.resolve()),
        "master_cache_bytes": master_out.stat().st_size,
        "master_cache_sha256": _sha256(master_out),
        "carrier_state": str(carrier_out.resolve()),
        "carrier_state_bytes": carrier_out.stat().st_size,
        "carrier_state_sha256": _sha256(carrier_out),
        "master_shape": list(masters.shape),
        "basis_shape": list(basis.shape),
        "coefficient_shape": list(coeff.shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--master-out", type=Path, required=True)
    parser.add_argument("--carrier-out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = extract(
        args.archive, args.raw, args.master_out, args.carrier_out
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
