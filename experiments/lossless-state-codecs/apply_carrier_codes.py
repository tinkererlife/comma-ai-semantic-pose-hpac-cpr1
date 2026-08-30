#!/usr/bin/env python3
"""Apply searched int12 carrier codes while preserving every other state byte."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from carrier_codec import (  # noqa: E402
    _zigzag_signed,
    decode_compact_carrier,
    encode_compact_carrier,
)
from model_bundle import (  # noqa: E402
    MODEL_LENGTH_MASK,
    decode_model_bundle,
    encode_model_bundle,
    parse_model_field,
)


FRAMES = 600
DIMENSIONS = 12
BASIS_COUNT = DIMENSIONS * 3 * 24 * 32


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _payload(path: Path) -> bytes:
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if len(infos) != 1 or infos[0].filename != "p":
            raise ValueError("archive must contain exactly one member named p")
        return archive.read("p")


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", allowZip64=False) as archive:
        info = zipfile.ZipInfo("p", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        archive.writestr(info, payload)


def apply_codes(
    source: Path,
    checkpoint: Path,
    output: Path,
    *,
    replace_basis: bool = False,
) -> dict:
    payload = _payload(source)
    model_bytes, flags = parse_model_field(struct.unpack_from("<I", payload)[0])
    token_offset = 4 + model_bytes
    bundle = decode_model_bundle(payload[4:token_offset], flags)
    source_basis_scales, source_basis_codes, source_coeff_scales, _ = decode_compact_carrier(
        bundle.carrier, BASIS_COUNT, FRAMES, DIMENSIONS
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    codes = state["coeff_codes"].detach().cpu().numpy().astype(np.int32)
    scales = state["coeff_scales"].detach().cpu().numpy().astype("<f4")
    if codes.shape != (FRAMES, DIMENSIONS):
        raise ValueError("searched coefficient codes have the wrong shape")
    if codes.min() < -2047 or codes.max() > 2047:
        raise ValueError("searched coefficient code exceeds the signed int12 deployment range")
    if not replace_basis and not np.array_equal(scales, source_coeff_scales):
        raise ValueError("coefficient scales changed during search")
    if replace_basis:
        if not all(key in state for key in ("basis_codes", "basis_scales")):
            raise ValueError("--replace-basis requires deployed basis codes and scales")
        basis_codes = state["basis_codes"].detach().cpu().numpy().astype(np.int8)
        basis_scales = state["basis_scales"].detach().cpu().numpy().astype("<f4")
        if basis_codes.size != BASIS_COUNT or basis_scales.shape != (DIMENSIONS,):
            raise ValueError("deployed basis state has the wrong shape")
    else:
        basis_codes = source_basis_codes
        basis_scales = source_basis_scales
    previous = np.vstack((np.zeros((1, DIMENSIONS), dtype=np.int32), codes[:-1]))
    delta = ((codes - previous + 2048) & 0xFFF) - 2048
    encoded = _zigzag_signed(delta, 12)
    canonical = encode_compact_carrier(
        basis_scales, basis_codes, scales, encoded
    )
    rebuilt_bundle, rebuilt_flags = encode_model_bundle(
        replace(bundle, carrier=canonical),
        semantic_codec=bundle.semantic_codec,
        carrier_codec=bundle.carrier_codec,
    )
    if len(rebuilt_bundle) > MODEL_LENGTH_MASK:
        raise ValueError("rebuilt model bundle exceeds CPR1 field")
    rebuilt_payload = (
        struct.pack("<I", len(rebuilt_bundle) | rebuilt_flags)
        + rebuilt_bundle
        + payload[token_offset:]
    )
    _write(output, rebuilt_payload)
    return {
        "source_archive_bytes": source.stat().st_size,
        "archive_bytes": output.stat().st_size,
        "archive_delta_bytes": output.stat().st_size - source.stat().st_size,
        "source_archive_sha256": _sha256(source.read_bytes()),
        "archive_sha256": _sha256(output.read_bytes()),
        "carrier_sha256": _sha256(canonical),
        "basis_replaced": replace_basis,
        "changed_basis_code_count": int(
            (basis_codes.reshape(-1) != source_basis_codes.reshape(-1)).sum()
        ),
        "changed_code_count": int((codes != state["initial_coeff_codes"].numpy()).sum())
        if "initial_coeff_codes" in state else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--replace-basis", action="store_true")
    args = parser.parse_args()
    result = apply_codes(
        args.archive,
        args.checkpoint,
        args.out,
        replace_basis=args.replace_basis,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
