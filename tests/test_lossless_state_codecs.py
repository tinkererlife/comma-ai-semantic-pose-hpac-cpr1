"""Exact storage-codec and CPR1 bundle regression tests."""

from __future__ import annotations

import hashlib
import importlib.util
import struct
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from cap1 import decode_cap1, encode_cap1  # noqa: E402
from carrier_codec import decode_compact_carrier  # noqa: E402
from model_bundle import (  # noqa: E402
    CAP1_MODEL_LENGTH_FLAG,
    RC64_MODEL_LENGTH_FLAG,
    WANS1_MODEL_LENGTH_FLAG,
    decode_model_bundle,
    parse_model_field,
)
from wans1 import decode_wans1, encode_wans1  # noqa: E402


def _canonical_bundle():
    with zipfile.ZipFile(ROOT / "artifacts/final/archive.zip") as archive:
        payload = archive.read("p")
    model_bytes, flags = parse_model_field(struct.unpack_from("<I", payload)[0])
    return payload, model_bytes, decode_model_bundle(payload[4:4 + model_bytes], flags)


def test_wans1_and_cap1_exact_public_vectors():
    _, _, bundle = _canonical_bundle()
    wans = encode_wans1(bundle.semantic)
    cap = encode_cap1(bundle.carrier)
    assert len(wans) == 36_051
    assert hashlib.sha256(wans).hexdigest() == (
        "d25c57ad89421bec43a93bed9f3d846a9eaecd3ce6f831729fe4dd47b6a2ef02"
    )
    assert len(cap) == 22_970
    assert hashlib.sha256(cap).hexdigest() == (
        "cc7912428c8443fbdb9f8df45dc817bff85179714f69e326e2ed3bf6e70cc9bf"
    )
    assert decode_wans1(wans) == bundle.semantic
    assert decode_cap1(cap) == bundle.carrier


def test_repacked_archive_restores_every_canonical_byte(tmp_path):
    module_path = ROOT / "experiments/lossless-state-codecs/repack_archive.py"
    spec = importlib.util.spec_from_file_location("repack_archive", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    source = ROOT / "artifacts/final/archive.zip"
    output = tmp_path / "repacked.zip"
    report = module.repack_archive(
        source, output, semantic_codec="wans1", carrier_codec="cap1"
    )
    source_payload, _, source_bundle = _canonical_bundle()
    with zipfile.ZipFile(output) as archive:
        output_payload = archive.read("p")
    model_bytes, flags = parse_model_field(struct.unpack_from("<I", output_payload)[0])
    rebuilt = decode_model_bundle(output_payload[4:4 + model_bytes], flags)
    assert flags & WANS1_MODEL_LENGTH_FLAG
    assert flags & CAP1_MODEL_LENGTH_FLAG
    assert not flags & RC64_MODEL_LENGTH_FLAG
    assert rebuilt.semantic == source_bundle.semantic
    assert rebuilt.carrier == source_bundle.carrier
    assert rebuilt.hpac == source_bundle.hpac
    source_token_offset = 4 + struct.unpack_from("<I", source_payload)[0]
    assert output_payload[4 + model_bytes:] == source_payload[source_token_offset:]
    assert report["parity"] == {
        "semantic": True,
        "carrier": True,
        "hpac": True,
        "tokens": True,
    }


def test_unchanged_searched_codes_rebuild_identical_archive(tmp_path):
    extract_spec = importlib.util.spec_from_file_location(
        "extract_pose_state", ROOT / "code/extract_deployed_pose_state.py"
    )
    assert extract_spec is not None and extract_spec.loader is not None
    extract_module = importlib.util.module_from_spec(extract_spec)
    extract_spec.loader.exec_module(extract_module)
    apply_spec = importlib.util.spec_from_file_location(
        "apply_carrier_codes",
        ROOT / "experiments/lossless-state-codecs/apply_carrier_codes.py",
    )
    assert apply_spec is not None and apply_spec.loader is not None
    apply_module = importlib.util.module_from_spec(apply_spec)
    apply_spec.loader.exec_module(apply_module)

    source = ROOT / "artifacts/final/archive.zip"
    _, _, bundle = _canonical_bundle()
    _, _, scales, encoded = decode_compact_carrier(
        bundle.carrier, 12 * 3 * 24 * 32, 600, 12
    )
    codes = extract_module._absolute_codes(encoded)
    checkpoint = tmp_path / "codes.pt"
    import torch

    torch.save({
        "coeff_codes": torch.from_numpy(codes),
        "coeff_scales": torch.from_numpy(scales.copy()),
        "initial_coeff_codes": torch.from_numpy(codes.copy()),
    }, checkpoint)
    output = tmp_path / "rebuilt.zip"
    report = apply_module.apply_codes(source, checkpoint, output)
    assert output.read_bytes() == source.read_bytes()
    assert report["changed_code_count"] == 0
