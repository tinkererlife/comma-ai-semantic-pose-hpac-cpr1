#!/usr/bin/env python3
"""Replace only the HPAC model and token stream in a confirmed submission."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import struct
import zipfile
from pathlib import Path


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
    parser.add_argument("--hpac", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--submission-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    with zipfile.ZipFile(args.base_archive) as archive:
        payload = archive.read("p")
    if len(payload) < 5:
        raise ValueError("base payload is truncated")
    old_models_bytes = struct.unpack_from("<I", payload)[0]
    old_models = lzma.decompress(payload[4:4 + old_models_bytes])
    if len(old_models) < 9:
        raise ValueError("base model payload is truncated")
    semantic_bytes, carrier_bytes = struct.unpack_from("<II", old_models)
    semantic_pose_bytes = 8 + semantic_bytes + carrier_bytes
    if semantic_pose_bytes >= len(old_models):
        raise ValueError("base model payload has no HPAC model")

    semantic_pose = old_models[:semantic_pose_bytes]
    hpac = lzma.decompress(args.hpac.read_bytes())
    models = lzma.compress(
        semantic_pose + hpac,
        format=lzma.FORMAT_XZ,
        filters=LZMA_FILTERS,
    )
    tokens = args.tokens.read_bytes()
    rebuilt = struct.pack("<I", len(models)) + models + tokens

    args.submission_dir.mkdir(parents=True, exist_ok=True)
    payload_path = args.submission_dir / "p"
    archive_path = args.submission_dir / "archive.zip"
    payload_path.write_bytes(rebuilt)
    write_deterministic_zip(archive_path, rebuilt)

    report = {
        "semantic_bytes": semantic_bytes,
        "carrier_bytes": carrier_bytes,
        "semantic_pose_bytes": semantic_pose_bytes,
        "old_combined_models_bytes": old_models_bytes,
        "new_combined_models_bytes": len(models),
        "hpac_raw_bytes": len(hpac),
        "token_bytes": len(tokens),
        "payload_bytes": len(rebuilt),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "semantic_pose_preserved_exact": (
            semantic_pose == old_models[:semantic_pose_bytes]
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
