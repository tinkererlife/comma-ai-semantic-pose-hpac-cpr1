#!/usr/bin/env python3
"""Build a submission with lossless temporal-delta carrier coefficients."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import struct
import zipfile
from pathlib import Path

import numpy as np


N = 600
CARRIER_DIM = 12
LZMA_FILTERS = [{
    "id": lzma.FILTER_LZMA2,
    "dict_size": 1 << 16,
    "lc": 0,
    "lp": 0,
    "pb": 0,
    "mode": lzma.MODE_NORMAL,
    "nice_len": 273,
    "mf": lzma.MF_BT4,
    "depth": 0,
}]


def unpack_signed_int12(blob: bytes, count: int) -> np.ndarray:
    packed = np.frombuffer(blob, dtype=np.uint8).astype(np.uint16)
    if len(packed) != ((count + 1) // 2) * 3:
        raise ValueError("unexpected packed int12 length")
    values = np.empty((len(packed) // 3) * 2, dtype=np.int16)
    values[0::2] = packed[0::3] | ((packed[1::3] & 0xF) << 8)
    values[1::2] = (packed[1::3] >> 4) | (packed[2::3] << 4)
    values[values >= 2048] -= 4096
    return values[:count]


def pack_unsigned_int12(values: np.ndarray) -> bytes:
    unsigned = (values.astype(np.int32, copy=False).reshape(-1) & 0xFFF).astype(
        np.uint16
    )
    if unsigned.size % 2:
        unsigned = np.pad(unsigned, (0, 1))
    first = unsigned[0::2]
    second = unsigned[1::2]
    packed = np.empty(first.size * 3, dtype=np.uint8)
    packed[0::3] = first & 0xFF
    packed[1::3] = ((first >> 8) & 0xF) | ((second & 0xF) << 4)
    packed[2::3] = second >> 4
    return packed.tobytes()


def encode_delta_zigzag(codes: np.ndarray) -> np.ndarray:
    codes = codes.astype(np.int32, copy=False).reshape(N, CARRIER_DIM)
    unsigned = codes & 0xFFF
    delta = unsigned.copy()
    delta[1:] = (unsigned[1:] - unsigned[:-1]) & 0xFFF
    signed_delta = np.where(delta >= 0x800, delta - 0x1000, delta)
    return ((signed_delta << 1) ^ (signed_delta >> 11)) & 0xFFF


def decode_delta_zigzag(encoded: np.ndarray) -> np.ndarray:
    encoded = encoded.astype(np.int32, copy=False).reshape(N, CARRIER_DIM) & 0xFFF
    delta = (encoded >> 1) ^ -(encoded & 1)
    restored = np.cumsum(delta, axis=0, dtype=np.int32) & 0xFFF
    return np.where(restored >= 0x800, restored - 0x1000, restored).astype(np.int16)


def write_deterministic_zip(output: Path, payload: bytes) -> None:
    with zipfile.ZipFile(output, "w", allowZip64=False) as archive:
        info = zipfile.ZipInfo("p", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        archive.writestr(info, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-archive", type=Path, required=True)
    parser.add_argument("--submission-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    with zipfile.ZipFile(args.base_archive) as archive:
        payload = archive.read("p")
    model_bytes = struct.unpack_from("<I", payload)[0]
    models = lzma.decompress(payload[4:4 + model_bytes])
    semantic_bytes, carrier_bytes = struct.unpack_from("<II", models)
    semantic_pose_bytes = 8 + semantic_bytes + carrier_bytes
    coefficient_bytes = (N * CARRIER_DIM // 2) * 3
    coefficient_start = semantic_pose_bytes - coefficient_bytes
    original_blob = models[coefficient_start:semantic_pose_bytes]
    original_codes = unpack_signed_int12(original_blob, N * CARRIER_DIM)
    encoded = encode_delta_zigzag(original_codes)
    transformed_blob = pack_unsigned_int12(encoded)
    restored = decode_delta_zigzag(
        unpack_signed_int12(transformed_blob, N * CARRIER_DIM)
    )
    if not np.array_equal(restored.reshape(-1), original_codes):
        raise ValueError("carrier coefficient transform did not round-trip")

    rebuilt_models = (
        models[:coefficient_start]
        + transformed_blob
        + models[semantic_pose_bytes:]
    )
    compressed_models = lzma.compress(
        rebuilt_models, format=lzma.FORMAT_XZ, filters=LZMA_FILTERS
    )
    rebuilt_payload = (
        struct.pack("<I", len(compressed_models))
        + compressed_models
        + payload[4 + model_bytes:]
    )
    args.submission_dir.mkdir(parents=True, exist_ok=True)
    payload_path = args.submission_dir / "p"
    archive_path = args.submission_dir / "archive.zip"
    payload_path.write_bytes(rebuilt_payload)
    write_deterministic_zip(archive_path, rebuilt_payload)

    report = {
        "base_combined_models_bytes": model_bytes,
        "combined_models_bytes": len(compressed_models),
        "coefficient_bytes": coefficient_bytes,
        "token_bytes": len(payload) - 4 - model_bytes,
        "payload_bytes": len(rebuilt_payload),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "coefficient_round_trip_exact": True,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
