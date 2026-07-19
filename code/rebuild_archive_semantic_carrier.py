#!/usr/bin/env python3
"""Replace semantic/carrier payloads while preserving charged HPAC and tokens."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import struct
import zipfile
from pathlib import Path

import torch

from build_submission_archive import LZMA_FILTERS, write_deterministic_zip
from pack_semantic_pose import pack_carrier, pack_semantic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-archive", type=Path, required=True)
    parser.add_argument("--semantic", type=Path)
    parser.add_argument("--carrier", type=Path, required=True)
    parser.add_argument("--basis-bits", type=int, choices=range(4, 9), required=True)
    parser.add_argument("--submission-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    with zipfile.ZipFile(args.base_archive) as archive:
        base_payload = archive.read("p")
    base_model_bytes = struct.unpack_from("<I", base_payload)[0]
    base_models = lzma.decompress(
        base_payload[4:4 + base_model_bytes]
    )
    semantic_bytes, carrier_bytes = struct.unpack_from("<II", base_models)
    semantic_start = 8
    carrier_start = semantic_start + semantic_bytes
    hpac_start = carrier_start + carrier_bytes
    if args.semantic is None:
        semantic_blob = base_models[semantic_start:carrier_start]
    else:
        semantic_checkpoint = torch.load(
            args.semantic, map_location="cpu", weights_only=False
        )
        semantic_blob, _ = pack_semantic(semantic_checkpoint)

    carrier_checkpoint = torch.load(
        args.carrier, map_location="cpu", weights_only=False
    )
    carrier_blob, _ = pack_carrier(
        carrier_checkpoint, basis_bits=args.basis_bits, coeff_bits=12
    )
    hpac_blob = base_models[hpac_start:]
    rebuilt_models = (
        struct.pack("<II", len(semantic_blob), len(carrier_blob))
        + semantic_blob
        + carrier_blob
        + hpac_blob
    )
    compressed_models = lzma.compress(
        rebuilt_models, format=lzma.FORMAT_XZ, filters=LZMA_FILTERS
    )
    tokens = base_payload[4 + base_model_bytes:]
    rebuilt_payload = (
        struct.pack("<I", len(compressed_models))
        + compressed_models
        + tokens
    )

    args.submission_dir.mkdir(parents=True, exist_ok=True)
    payload_path = args.submission_dir / "p"
    archive_path = args.submission_dir / "archive.zip"
    payload_path.write_bytes(rebuilt_payload)
    write_deterministic_zip(archive_path, [("p", payload_path)])

    report = {
        "base_archive": str(args.base_archive.resolve()),
        "base_model_bytes": base_model_bytes,
        "base_semantic_bytes": semantic_bytes,
        "base_carrier_bytes": carrier_bytes,
        "semantic_bytes": len(semantic_blob),
        "carrier_bytes": len(carrier_blob),
        "carrier_basis_bits": args.basis_bits,
        "hpac_bytes": len(hpac_blob),
        "combined_models_bytes": len(compressed_models),
        "token_bytes": len(tokens),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "preserved_hpac": hpac_blob == base_models[hpac_start:],
        "preserved_tokens": tokens == base_payload[4 + base_model_bytes:],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
