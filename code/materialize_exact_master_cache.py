#!/usr/bin/env python3
"""Cache exact deployed master frames from an inflated challenge video."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


FRAMES = 600
CAMERA_SHAPE = (874, 1164, 3)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    expected_bytes = 2 * FRAMES * int(np.prod(CAMERA_SHAPE))
    if args.raw.stat().st_size != expected_bytes:
        raise ValueError(
            f"inflated raw size mismatch: {args.raw.stat().st_size} != {expected_bytes}"
        )
    frames = np.memmap(
        args.raw,
        mode="r",
        dtype=np.uint8,
        shape=(2 * FRAMES,) + CAMERA_SHAPE,
    )
    masters = torch.from_numpy(np.asarray(frames[1::2]).copy()).permute(0, 3, 1, 2)
    archive_sha256 = _sha256(args.archive)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"masters": masters, "source_archive_sha256": archive_sha256},
        args.out,
    )
    result = {
        "archive_sha256": archive_sha256,
        "master_cache": str(args.out.resolve()),
        "master_cache_bytes": args.out.stat().st_size,
        "master_cache_sha256": _sha256(args.out),
        "master_shape": list(masters.shape),
        "raw_sha256": _sha256(args.raw),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
