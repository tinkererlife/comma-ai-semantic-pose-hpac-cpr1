#!/usr/bin/env python3
"""Replace only F24S HPAC bytes and its exact RC64 token stream."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import lzma
import sys
from pathlib import Path


def load_renderer(path: Path):
    spec = importlib.util.spec_from_file_location("_f24_hpac_renderer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load renderer runtime: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-archive", type=Path, required=True)
    parser.add_argument("--ihs1", type=Path, required=True)
    parser.add_argument("--token-stream", type=Path)
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--experiment-book-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.experiment_book_root / "src"))
    from cpr1_sub4.baseline import BaselinePayload, encode_legacy_w4
    from cpr1_sub4.entropy.renderer_weight_codec import decode_wans1
    from cpr1_sub4.f14_renderer import F14_FILTERS
    from cpr1_sub4.ihs2 import encode_ihs2_v3, layout_from_runtime, materialize_ihs1
    from cpr1_sub4.residual_archive import (
        FIXED_SCHEMA,
        build_residual_archive_bytes,
        read_residual_archive,
    )

    parts = read_residual_archive(args.base_archive)
    if not parts.hpac_blob.startswith(b"IHS2\x03"):
        raise ValueError("base archive does not contain an IHS2-v3 HPAC")
    raw = args.ihs1.read_bytes()
    if raw.startswith(b"\xfd7zXZ\x00"):
        raw = lzma.decompress(raw)
    if not raw.startswith(b"IHS1"):
        raise ValueError("--ihs1 is neither raw nor XZ-compressed IHS1")

    renderer = load_renderer(args.renderer)
    flags = parts.hpac_blob[5]
    candidate_hpac = encode_ihs2_v3(
        raw,
        layout_from_runtime(renderer),
        frame_format=flags & 7,
        pack_exponents=bool(flags & 8),
        tighten_rows=bool(flags & 16),
        pack_biases=bool(flags & 32),
    )
    if materialize_ihs1(candidate_hpac, renderer) != raw:
        raise RuntimeError("IHS2 candidate does not reconstruct the requested IHS1")
    if len(candidate_hpac) != len(parts.hpac_blob):
        raise ValueError(
            "candidate HPAC cannot use fixed F24S schema: "
            f"{len(candidate_hpac)} != {len(parts.hpac_blob)} bytes"
        )

    stream = (
        args.token_stream.read_bytes()
        if args.token_stream is not None
        else parts.token_stream
    )
    records = decode_wans1(parts.semantic_blob)
    baseline = BaselinePayload(
        archive_path=args.base_archive,
        semantic_blob=encode_legacy_w4(records),
        carrier_blob=parts.carrier_blob,
        hpac_blob=parts.hpac_blob,
        token_stream=parts.token_stream,
        records=records,
    )
    archive, ledger = build_residual_archive_bytes(
        baseline,
        parts.table,
        stream,
        FIXED_SCHEMA,
        hpac_blob=candidate_hpac,
        semantic_blob=parts.semantic_blob,
        carrier_blob=parts.carrier_blob,
        lzma_filters=F14_FILTERS,
        fixed_wans_ar1_rc64_schema=True,
        model_compression="raw",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(archive)

    restored = read_residual_archive(args.out)
    if (
        restored.semantic_blob != parts.semantic_blob
        or restored.carrier_blob != parts.carrier_blob
        or restored.hpac_blob != candidate_hpac
        or restored.token_stream != stream
    ):
        raise RuntimeError("rebuilt F24S component parity failed")

    result = {
        "base_archive_bytes": args.base_archive.stat().st_size,
        "archive_bytes": len(archive),
        "archive_delta_bytes": len(archive) - args.base_archive.stat().st_size,
        "archive_sha256": hashlib.sha256(archive).hexdigest(),
        "ihs1_bytes": len(raw),
        "ihs1_sha256": hashlib.sha256(raw).hexdigest(),
        "ihs2_bytes": len(candidate_hpac),
        "ihs2_sha256": hashlib.sha256(candidate_hpac).hexdigest(),
        "token_stream_bytes": len(stream),
        "token_stream_sha256": hashlib.sha256(stream).hexdigest(),
        "model_container_schema": ledger["model_container_schema"],
        "component_parity_verified": True,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
