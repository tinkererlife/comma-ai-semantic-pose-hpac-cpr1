#!/usr/bin/env python3
"""Repack CPR1 semantic/carrier state without changing decoded bytes."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import struct
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from model_bundle import (  # noqa: E402
    MODEL_LENGTH_MASK,
    decode_model_bundle,
    encode_model_bundle,
    parse_model_field,
)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_payload(path: Path) -> bytes:
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if len(infos) != 1 or infos[0].filename != "p":
            raise ValueError("archive must contain exactly one member named p")
        if infos[0].compress_type != zipfile.ZIP_STORED:
            raise ValueError("archive member p must be stored")
        return archive.read("p")


def _write_payload(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", allowZip64=False) as archive:
        info = zipfile.ZipInfo("p", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        archive.writestr(info, payload)


def repack_archive(
    source: Path,
    output: Path,
    *,
    semantic_codec: str,
    carrier_codec: str,
    carrier_from_archive: Path | None = None,
    semantic_from_archive: Path | None = None,
) -> dict:
    payload = _read_payload(source)
    if len(payload) < 4:
        raise ValueError("CPR1 payload is truncated")
    model_bytes, flags = parse_model_field(struct.unpack_from("<I", payload)[0])
    token_offset = 4 + model_bytes
    if token_offset > len(payload):
        raise ValueError("compressed model length exceeds the payload")
    compressed = payload[4:token_offset]
    tokens = payload[token_offset:]
    bundle = decode_model_bundle(compressed, flags)
    original_carrier_sha256 = _sha256(bundle.carrier)
    original_semantic_sha256 = _sha256(bundle.semantic)
    carrier_source_sha256 = None
    semantic_source_sha256 = None
    if semantic_from_archive is not None:
        semantic_payload = _read_payload(semantic_from_archive)
        if len(semantic_payload) < 4:
            raise ValueError("semantic-source CPR1 payload is truncated")
        source_bytes, source_flags = parse_model_field(
            struct.unpack_from("<I", semantic_payload)[0]
        )
        source_end = 4 + source_bytes
        if source_end > len(semantic_payload):
            raise ValueError("semantic-source model length exceeds its payload")
        semantic_source = decode_model_bundle(
            semantic_payload[4:source_end], source_flags
        )
        bundle = replace(bundle, semantic=semantic_source.semantic)
        semantic_source_sha256 = _sha256(semantic_from_archive.read_bytes())
    if carrier_from_archive is not None:
        carrier_payload = _read_payload(carrier_from_archive)
        if len(carrier_payload) < 4:
            raise ValueError("carrier-source CPR1 payload is truncated")
        source_bytes, source_flags = parse_model_field(
            struct.unpack_from("<I", carrier_payload)[0]
        )
        source_end = 4 + source_bytes
        if source_end > len(carrier_payload):
            raise ValueError("carrier-source model length exceeds its payload")
        carrier_source = decode_model_bundle(
            carrier_payload[4:source_end], source_flags
        )
        bundle = replace(bundle, carrier=carrier_source.carrier)
        carrier_source_sha256 = _sha256(carrier_from_archive.read_bytes())
    rebuilt, rebuilt_flags = encode_model_bundle(
        bundle, semantic_codec=semantic_codec, carrier_codec=carrier_codec
    )
    decoded = decode_model_bundle(rebuilt, rebuilt_flags)
    parity = {
        "semantic": decoded.semantic == bundle.semantic,
        "carrier": decoded.carrier == bundle.carrier,
        "hpac": decoded.hpac == bundle.hpac,
        "tokens": tokens == payload[token_offset:],
    }
    if not all(parity.values()):
        raise RuntimeError(f"lossless state parity failed: {parity}")
    if len(rebuilt) > MODEL_LENGTH_MASK:
        raise ValueError("compressed model bundle exceeds CPR1 field")
    result_payload = struct.pack("<I", len(rebuilt) | rebuilt_flags) + rebuilt + tokens
    _write_payload(output, result_payload)
    return {
        "source": str(source.resolve()),
        "output": str(output.resolve()),
        "source_archive_bytes": source.stat().st_size,
        "archive_bytes": output.stat().st_size,
        "saved_bytes": source.stat().st_size - output.stat().st_size,
        "semantic_codec": semantic_codec,
        "carrier_codec": carrier_codec,
        "carrier_from_archive": (
            str(carrier_from_archive.resolve()) if carrier_from_archive else None
        ),
        "carrier_source_archive_sha256": carrier_source_sha256,
        "original_carrier_sha256": original_carrier_sha256,
        "semantic_from_archive": (
            str(semantic_from_archive.resolve()) if semantic_from_archive else None
        ),
        "semantic_source_archive_sha256": semantic_source_sha256,
        "original_semantic_sha256": original_semantic_sha256,
        "token_codec": bundle.token_codec,
        "source_model_bytes": len(compressed),
        "model_bytes": len(rebuilt),
        "token_bytes": len(tokens),
        "decoded_bytes": {
            "semantic": len(bundle.semantic),
            "carrier": len(bundle.carrier),
            "hpac": len(bundle.hpac),
        },
        "decoded_sha256": {
            "semantic": _sha256(bundle.semantic),
            "carrier": _sha256(bundle.carrier),
            "hpac": _sha256(bundle.hpac),
            "tokens": _sha256(tokens),
        },
        "parity": parity,
        "archive_sha256": _sha256(output.read_bytes()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--semantic-codec", choices=("legacy", "wans1"), default="wans1"
    )
    parser.add_argument(
        "--carrier-codec", choices=("legacy", "cap1"), default="cap1"
    )
    parser.add_argument(
        "--carrier-from-archive",
        type=Path,
        help="replace only the canonical carrier using another CPR1 archive",
    )
    parser.add_argument(
        "--semantic-from-archive",
        type=Path,
        help="replace only the canonical semantic renderer from another archive",
    )
    args = parser.parse_args()
    result = repack_archive(
        args.archive,
        args.out,
        semantic_codec=args.semantic_codec,
        carrier_codec=args.carrier_codec,
        carrier_from_archive=args.carrier_from_archive,
        semantic_from_archive=args.semantic_from_archive,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
