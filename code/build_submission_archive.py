#!/usr/bin/env python3
"""Build the exact charged semantic-pose submission archive deterministically."""

from __future__ import annotations

import argparse
import json
import lzma
import math
import zipfile
from pathlib import Path

import numpy as np
import torch

from pack_semantic_pose import pack_carrier, pack_semantic


ORIGINAL_BYTES = 37_545_489
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


def write_semantic_pose(
    semantic_path: Path, carrier_path: Path, output: Path, basis_bits: int = 8
):
    semantic_checkpoint = torch.load(
        semantic_path, map_location="cpu", weights_only=False
    )
    carrier_checkpoint = torch.load(
        carrier_path, map_location="cpu", weights_only=False
    )
    semantic_blob, _ = pack_semantic(semantic_checkpoint)
    carrier_blob, _ = pack_carrier(
        carrier_checkpoint, basis_bits=basis_bits, coeff_bits=12
    )
    header = np.asarray(
        [len(semantic_blob), len(carrier_blob)], dtype=np.uint32
    ).tobytes()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(lzma.compress(
        header + semantic_blob + carrier_blob,
        format=lzma.FORMAT_XZ,
        filters=LZMA_FILTERS,
    ))
    return semantic_checkpoint, carrier_checkpoint, {
        "semantic_bytes": len(semantic_blob),
        "carrier_bytes": len(carrier_blob),
        "carrier_basis_bits": basis_bits,
        "raw_semantic_pose_bytes": len(header) + len(semantic_blob) + len(carrier_blob),
        "lzma_semantic_pose_bytes": output.stat().st_size,
    }


def write_deterministic_zip(output: Path, members: list[tuple[str, Path]]):
    with zipfile.ZipFile(output, "w", allowZip64=False) as archive:
        for name, path in members:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--carrier", type=Path, required=True)
    parser.add_argument("--hpac", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--basis-bits", type=int, choices=range(4, 9), default=8)
    parser.add_argument("--submission-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--projected-seg", type=float)
    parser.add_argument("--projected-pose", type=float)
    args = parser.parse_args()

    args.submission_dir.mkdir(parents=True, exist_ok=True)
    semantic_pose_path = args.submission_dir / "semantic_pose.bin.xz"
    semantic_checkpoint, carrier_checkpoint, result = write_semantic_pose(
        args.semantic, args.carrier, semantic_pose_path, args.basis_bits
    )
    hpac_raw = lzma.decompress(args.hpac.read_bytes())
    semantic_pose_raw = lzma.decompress(semantic_pose_path.read_bytes())
    combined_models = lzma.compress(
        semantic_pose_raw + hpac_raw,
        format=lzma.FORMAT_XZ,
        filters=LZMA_FILTERS,
    )
    tokens = args.tokens.read_bytes()
    payload_path = args.submission_dir / "p"
    payload_path.write_bytes(
        np.asarray([len(combined_models)], dtype=np.uint32).tobytes()
        + combined_models
        + tokens
    )
    archive_path = args.submission_dir / "archive.zip"
    write_deterministic_zip(archive_path, [("p", payload_path)])

    seg = (
        args.projected_seg
        if args.projected_seg is not None
        else float(semantic_checkpoint["result"]["quantized_exact_seg"])
    )
    carrier_result = carrier_checkpoint["result"]
    pose_record = carrier_result.get(
        "quantized_basis_coeff",
        carrier_result.get("quantized_basis_int8_coeff"),
    )
    if pose_record is None:
        raise ValueError("carrier checkpoint has no quantized pose summary")
    pose = (
        args.projected_pose
        if args.projected_pose is not None
        else float(pose_record["mean"])
    )
    archive_bytes = archive_path.stat().st_size
    score = 100.0 * seg + math.sqrt(10.0 * pose)
    score += 25.0 * archive_bytes / ORIGINAL_BYTES
    result.update({
        "hpac_model_bytes": args.hpac.stat().st_size,
        "combined_models_bytes": len(combined_models),
        "token_bytes": len(tokens),
        "payload_bytes": payload_path.stat().st_size,
        "archive_bytes": archive_bytes,
        "projected_seg": seg,
        "projected_pose": pose,
        "projected_score": score,
        "landslide_threshold": 0.1785487,
        "projected_landslide": score < 0.1785487,
    })
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
