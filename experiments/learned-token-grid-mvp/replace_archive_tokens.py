#!/usr/bin/env python3
"""Build a CPR1 variant while preserving every unchanged payload byte."""

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
RC64_MODEL_LENGTH_FLAG = 1 << 31


def replace_token_stream(
    base_archive: Path,
    tokens: bytes,
    output: Path,
    semantic_blob: bytes | None = None,
    token_codec: str = "range32",
) -> dict:
    with zipfile.ZipFile(base_archive) as archive:
        infos = archive.infolist()
        if len(infos) != 1 or infos[0].filename != "p":
            raise ValueError("base archive must contain one member named p")
        payload = archive.read("p")
    if len(payload) < 4:
        raise ValueError("truncated CPR1 payload")
    model_field = struct.unpack_from("<I", payload)[0]
    model_bytes = model_field & ~RC64_MODEL_LENGTH_FLAG
    token_offset = 4 + model_bytes
    if token_offset > len(payload):
        raise ValueError("model length exceeds CPR1 payload")
    if not tokens:
        raise ValueError("token stream must not be empty")
    if token_codec == "range32" and len(tokens) % 4:
        raise ValueError("range32 token stream must contain uint32 words")

    output_model_field = model_bytes | (
        RC64_MODEL_LENGTH_FLAG if token_codec == "rc64" else 0
    )
    model_prefix = struct.pack("<I", output_model_field) + payload[4:token_offset]
    preserved_model_bytes = None
    if semantic_blob is not None:
        models_raw = lzma.decompress(payload[4:token_offset])
        if len(models_raw) < 8:
            raise ValueError("truncated CPR1 model payload")
        semantic_bytes, carrier_bytes = struct.unpack_from("<II", models_raw)
        preserved_offset = 8 + semantic_bytes
        if preserved_offset + carrier_bytes > len(models_raw):
            raise ValueError("semantic or carrier length exceeds model payload")
        preserved = models_raw[preserved_offset:]
        rebuilt_models_raw = (
            struct.pack("<II", len(semantic_blob), carrier_bytes)
            + semantic_blob
            + preserved
        )
        rebuilt_models = lzma.compress(
            rebuilt_models_raw, format=lzma.FORMAT_XZ, filters=LZMA_FILTERS
        )
        output_model_field = len(rebuilt_models) | (
            RC64_MODEL_LENGTH_FLAG if token_codec == "rc64" else 0
        )
        model_prefix = struct.pack("<I", output_model_field) + rebuilt_models
        preserved_model_bytes = len(preserved)

    rebuilt = model_prefix + tokens
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
        "learned_model_prefix_bytes": len(model_prefix),
        "base_token_bytes": len(payload) - token_offset,
        "learned_token_bytes": len(tokens),
        "token_codec": token_codec,
        "semantic_replaced": semantic_blob is not None,
        "semantic_bytes": len(semantic_blob) if semantic_blob is not None else None,
        "preserved_model_bytes": preserved_model_bytes,
        "archive_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-archive", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--semantic-blob", type=Path)
    parser.add_argument(
        "--token-codec", choices=("range32", "rc64"), default="range32"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = replace_token_stream(
        args.base_archive,
        args.tokens.read_bytes(),
        args.out,
        args.semantic_blob.read_bytes() if args.semantic_blob else None,
        args.token_codec,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
