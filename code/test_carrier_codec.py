#!/usr/bin/env python3
"""Tests for the CPR1 carrier and the frozen archive repack."""

from __future__ import annotations

import hashlib
import importlib.util
import lzma
import os
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import carrier_codec  # noqa: E402
import repack_carrier  # noqa: E402


EXPECTED_ARCHIVE_BYTES = 191_052
EXPECTED_ARCHIVE_SHA256 = (
    "0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd"
)
EXPECTED_CARRIER_SHA256 = (
    "a05d0985ca5a8d5110bd5bf5be39f238c6f89640b8a8bb888a3e1269bdf636e4"
)
EXPECTED_SEMANTIC_STATE_SHA256 = (
    "de0e5fc616b75eb0bdb55528aa61a7405eebbd3a81064f513adfbb69b33105ba"
)
EXPECTED_BASIS_TENSOR_SHA256 = (
    "7a6576e991a068e084ffc12f6377b9bfcc00fd2529eb8df27c424921f3c3933b"
)
EXPECTED_COEFFICIENT_TENSOR_SHA256 = (
    "dee2587ec99eea45e76ebd68eaaad3e3ae52d51e0a9b673103be8d4128e07ca8"
)


def sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    return sha256(value.contiguous().numpy().tobytes())


def semantic_state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in state.items():
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(value.contiguous().numpy().tobytes())
    return digest.hexdigest()


def synthetic_carrier() -> tuple:
    dimensions = 12
    frames = 37
    basis_codes = np.resize(np.arange(-16, 16, dtype=np.int32), 1_280)
    rng = np.random.default_rng(20260717)
    rng.shuffle(basis_codes)
    basis_scales = np.linspace(0.01, 0.12, dimensions, dtype=np.float32)
    coefficient_scales = np.linspace(
        0.001,
        0.012,
        dimensions,
        dtype=np.float32,
    )
    coefficients = rng.integers(
        0,
        1 << carrier_codec.COEFF_BITS,
        size=(frames, dimensions),
        dtype=np.int32,
    )
    coefficients[0, :4] = [0, 1, 0xFFE, 0xFFF]
    return (
        basis_scales,
        basis_codes,
        coefficient_scales,
        coefficients,
    )


def test_compact_carrier_round_trip_is_deterministic():
    values = synthetic_carrier()
    first = carrier_codec.encode_compact_carrier(*values)
    second = carrier_codec.encode_compact_carrier(*values)
    assert first == second
    decoded = carrier_codec.decode_compact_carrier(
        first,
        basis_count=values[1].size,
        frames=values[3].shape[0],
        dimensions=values[3].shape[1],
    )
    for actual, expected in zip(decoded, values, strict=True):
        assert np.array_equal(actual, expected)


def test_randomized_compact_carriers_round_trip_exactly():
    rng = np.random.default_rng(0xC0DEC0DE)
    for _ in range(200):
        dimensions = int(rng.integers(1, 20))
        frames = int(rng.integers(1, 80))
        basis_count = int(rng.integers(2, 4_096))
        basis = rng.integers(-16, 16, size=basis_count, dtype=np.int32)
        if rng.integers(7) == 0:
            basis.fill(int(rng.integers(-16, 16)))
        values = (
            rng.random(dimensions, dtype=np.float32),
            basis,
            rng.random(dimensions, dtype=np.float32),
            rng.integers(
                0,
                1 << carrier_codec.COEFF_BITS,
                size=(frames, dimensions),
                dtype=np.int32,
            ),
        )
        blob = carrier_codec.encode_compact_carrier(*values)
        decoded = carrier_codec.decode_compact_carrier(
            blob,
            basis_count=basis_count,
            frames=frames,
            dimensions=dimensions,
        )
        for actual, expected in zip(decoded, values, strict=True):
            assert np.array_equal(actual, expected)


@pytest.mark.parametrize("cut", [0, 1, 11, 12, 31, -1])
def test_compact_carrier_rejects_truncation(cut):
    values = synthetic_carrier()
    blob = carrier_codec.encode_compact_carrier(*values)
    truncated = blob[:cut] if cut >= 0 else blob[:-1]
    with pytest.raises(ValueError):
        carrier_codec.decode_compact_carrier(
            truncated,
            basis_count=values[1].size,
            frames=values[3].shape[0],
            dimensions=values[3].shape[1],
        )


def test_compact_carrier_rejects_bad_magic_and_bad_rice_parameter():
    values = synthetic_carrier()
    blob = bytearray(carrier_codec.encode_compact_carrier(*values))
    blob[:4] = b"BAD!"
    with pytest.raises(ValueError, match="magic"):
        carrier_codec.decode_compact_carrier(
            blob,
            basis_count=values[1].size,
            frames=values[3].shape[0],
            dimensions=values[3].shape[1],
        )

    blob = bytearray(carrier_codec.encode_compact_carrier(*values))
    dimensions = values[3].shape[1]
    rice_table_offset = (
        carrier_codec.HEADER.size
        + 2 * dimensions * 4
        + carrier_codec.ALPHABET_SIZE
    )
    blob[rice_table_offset] = carrier_codec.COEFF_BITS
    with pytest.raises(ValueError, match="Rice parameter"):
        carrier_codec.decode_compact_carrier(
            blob,
            basis_count=values[1].size,
            frames=values[3].shape[0],
            dimensions=dimensions,
        )


def test_compact_carrier_rejects_oversubscribed_huffman_table():
    values = synthetic_carrier()
    blob = bytearray(carrier_codec.encode_compact_carrier(*values))
    dimensions = values[3].shape[1]
    lengths_offset = carrier_codec.HEADER.size + 2 * dimensions * 4
    blob[lengths_offset:lengths_offset + carrier_codec.ALPHABET_SIZE] = (
        bytes([1, 1, 1]) + bytes(carrier_codec.ALPHABET_SIZE - 3)
    )
    with pytest.raises(ValueError, match="oversubscribed"):
        carrier_codec.decode_compact_carrier(
            blob,
            basis_count=values[1].size,
            frames=values[3].shape[0],
            dimensions=dimensions,
        )


def test_compact_carrier_rejects_incomplete_huffman_table():
    values = synthetic_carrier()
    blob = bytearray(carrier_codec.encode_compact_carrier(*values))
    dimensions = values[3].shape[1]
    lengths_offset = carrier_codec.HEADER.size + 2 * dimensions * 4
    blob[lengths_offset:lengths_offset + carrier_codec.ALPHABET_SIZE] = (
        bytes([2, 2]) + bytes(carrier_codec.ALPHABET_SIZE - 2)
    )
    with pytest.raises(ValueError, match="incomplete"):
        carrier_codec.decode_compact_carrier(
            blob,
            basis_count=values[1].size,
            frames=values[3].shape[0],
            dimensions=dimensions,
        )


def test_bit_payload_rejects_nonzero_padding_and_surplus_declared_bits():
    values = synthetic_carrier()
    encoded = carrier_codec.encode_compact_carrier(*values)
    dimensions = values[3].shape[1]
    basis_bits, coefficient_bits = struct.unpack_from("<II", encoded, 4)
    coefficient_bytes = (coefficient_bits + 7) // 8
    assert coefficient_bits % 8

    blob = bytearray(encoded)
    blob[-1] |= 1
    with pytest.raises(ValueError, match="nonzero padding"):
        carrier_codec.decode_compact_carrier(
            blob,
            basis_count=values[1].size,
            frames=values[3].shape[0],
            dimensions=dimensions,
        )

    assert (coefficient_bits + 1 + 7) // 8 == coefficient_bytes
    blob = bytearray(encoded)
    carrier_codec.HEADER.pack_into(
        blob,
        0,
        carrier_codec.MAGIC,
        basis_bits,
        coefficient_bits + 1,
    )
    with pytest.raises(ValueError, match="surplus declared bits"):
        carrier_codec.decode_compact_carrier(
            blob,
            basis_count=values[1].size,
            frames=values[3].shape[0],
            dimensions=dimensions,
        )


def test_rice_decoder_rejects_unary_overflow_and_surplus_bits():
    with pytest.raises(ValueError, match="exceeds 12-bit range"):
        carrier_codec._decode_rice(
            np.array([11], dtype=np.uint8),
            bytes([0b00100000]),
            bit_count=3,
            frames=1,
            dimensions=1,
        )

    with pytest.raises(ValueError, match="surplus declared bits"):
        carrier_codec._decode_rice(
            np.array([0], dtype=np.uint8),
            bytes([0b11000000]),
            bit_count=2,
            frames=1,
            dimensions=1,
        )


def load_semantic_pose(archive_path: Path) -> bytes:
    with zipfile.ZipFile(archive_path) as archive:
        payload = archive.read("p")
    compressed_bytes = struct.unpack_from("<I", payload)[0]
    models = lzma.decompress(payload[4:4 + compressed_bytes])
    semantic_bytes, carrier_bytes = struct.unpack_from("<II", models)
    return models[:8 + semantic_bytes + carrier_bytes]


@pytest.mark.skipif(
    "SEMANTIC_POSE_SOURCE_ARCHIVE" not in os.environ,
    reason="set SEMANTIC_POSE_SOURCE_ARCHIVE for the frozen archive golden",
)
def test_frozen_archive_repack_and_decoded_models_are_exact(tmp_path):
    source = Path(os.environ["SEMANTIC_POSE_SOURCE_ARCHIVE"])
    output = tmp_path / "archive.zip"
    report = repack_carrier.repack(source, output)
    assert output.stat().st_size == EXPECTED_ARCHIVE_BYTES
    assert sha256(output.read_bytes()) == EXPECTED_ARCHIVE_SHA256
    assert report["payload"]["carrier_sha256"] == EXPECTED_CARRIER_SHA256
    assert report["payload"]["lossless_carrier_roundtrip"]
    generic_output = tmp_path / "generic.zip"
    generic_report = repack_carrier.repack(
        source,
        generic_output,
        require_canonical_source=False,
    )
    assert generic_output.read_bytes() == output.read_bytes()
    assert generic_report["canonical_source_required"] is False
    assert "projection_from_displayed_metrics" not in generic_report

    spec = importlib.util.spec_from_file_location(
        "semantic_pose_landslide_selfcompress_inflate",
        HERE / "inflate.py",
    )
    assert spec is not None and spec.loader is not None
    inflate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inflate)
    old_semantic, old_basis, old_coefficients = inflate.unpack_semantic_pose(
        load_semantic_pose(source)
    )
    new_semantic, new_basis, new_coefficients = inflate.unpack_semantic_pose(
        load_semantic_pose(output)
    )
    assert old_semantic.state_dict().keys() == new_semantic.state_dict().keys()
    for name, old_value in old_semantic.state_dict().items():
        assert torch.equal(old_value, new_semantic.state_dict()[name]), name
    assert torch.equal(old_basis, new_basis)
    assert torch.equal(old_coefficients, new_coefficients)
    assert (
        semantic_state_sha256(new_semantic.state_dict())
        == EXPECTED_SEMANTIC_STATE_SHA256
    )
    assert tensor_sha256(new_basis) == EXPECTED_BASIS_TENSOR_SHA256
    assert (
        tensor_sha256(new_coefficients)
        == EXPECTED_COEFFICIENT_TENSOR_SHA256
    )
