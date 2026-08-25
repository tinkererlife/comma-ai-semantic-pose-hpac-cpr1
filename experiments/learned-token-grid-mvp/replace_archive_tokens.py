#!/usr/bin/env python3
"""Build a CPR1 variant by replacing only its arithmetic token stream."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import zipfile
from pathlib import Path


def replace_token_stream(base_archive: Path, tokens: bytes, output: Path) -> dict:
    with zipfile.ZipFile(base_archive) as archive:
        infos = archive.infolist()
        if len(infos) != 1 or infos[0].filename != "p":
            raise ValueError("base archive must contain one member named p")
        payload = archive.read("p")
    if len(payload) < 4:
        raise ValueError("truncated CPR1 payload")
    model_bytes = struct.unpack_from("<I", payload)[0]
    token_offset = 4 + model_bytes
    if token_offset > len(payload):
        raise ValueError("model length exceeds CPR1 payload")
    if not tokens or len(tokens) % 4:
        raise ValueError("range-coded token stream must contain uint32 words")

    rebuilt = payload[:token_offset] + tokens
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", allowZip64=False) as archive:
        info = zipfile.ZipInfo("p", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        archive.writestr(info, rebuilt)
    return {
        "base_archive_bytes": base_archive.stat().st_size,
        "archive_bytes": output.stat().st_size,
        "model_prefix_bytes": token_offset,
        "base_token_bytes": len(payload) - token_offset,
        "learned_token_bytes": len(tokens),
        "archive_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-archive", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = replace_token_stream(args.base_archive, args.tokens.read_bytes(), args.out)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
