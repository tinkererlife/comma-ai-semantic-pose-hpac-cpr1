"""Exact CAP1 AR(1)+bias/Rice representation for the CPR1 pose carrier.

Adapted from codexblack's public PR #135 ExperimentBook implementation.  CAP1
changes only storage: decoding restores the canonical ``CPR1`` carrier bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
import struct

import numpy as np

from carrier_codec import (
    MAGIC as CPR1_MAGIC,
    _decode_rice,
    _encode_rice,
    _unzigzag_unsigned,
    _zigzag_signed,
)


MAGIC = b"CAP1"
VERSION = 1
HEADER_BYTES = 14
Q8_MIN = -512
Q8_MAX = 512
BIAS_MIN = -16
BIAS_MAX = 16


class Cap1Error(ValueError):
    """A CAP1 stream or reconstructed CPR1 carrier is invalid."""


def _u24(value: int) -> bytes:
    if not 0 < value < 1 << 24:
        raise Cap1Error("CAP1 bit count does not fit u24")
    return value.to_bytes(3, "little")


def _read_u24(raw: bytes) -> int:
    if len(raw) != 3:
        raise Cap1Error("truncated CAP1 u24")
    return int.from_bytes(raw, "little")


def _signed_mod(values: np.ndarray) -> np.ndarray:
    return ((np.asarray(values, dtype=np.int64) + 2048) & 0xFFF).astype(np.int32) - 2048


def _round_q8(values: np.ndarray, factors: np.ndarray) -> np.ndarray:
    products = np.asarray(values, dtype=np.int64) * np.asarray(factors, dtype=np.int64)
    return np.where(
        products >= 0,
        (products + 128) // 256,
        -((-products + 128) // 256),
    ).astype(np.int32)


@dataclass(frozen=True)
class _Ar1Bias:
    factors_q8: np.ndarray
    biases: np.ndarray

    def __post_init__(self) -> None:
        factors = np.asarray(self.factors_q8)
        biases = np.asarray(self.biases)
        if factors.ndim != 1 or factors.shape != biases.shape or not factors.size:
            raise Cap1Error("CAP1 AR vectors must be equal and nonempty")
        if np.any(factors < Q8_MIN) or np.any(factors > Q8_MAX):
            raise Cap1Error("CAP1 AR factor outside the fixed schema")
        if np.any(biases < BIAS_MIN) or np.any(biases > BIAS_MAX):
            raise Cap1Error("CAP1 bias outside the fixed schema")


def _require_coefficients(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 2 or result.shape[0] < 2 or not np.issubdtype(result.dtype, np.integer):
        raise Cap1Error("CAP1 coefficients must be a two-dimensional integer matrix")
    result = result.astype(np.int32, copy=False)
    if np.any(result < -2048) or np.any(result > 2047):
        raise Cap1Error("CAP1 coefficient exceeds signed int12")
    return result


def _residuals(coefficients: np.ndarray, model: _Ar1Bias) -> np.ndarray:
    values = _require_coefficients(coefficients)
    if values.shape[1] != len(model.factors_q8):
        raise Cap1Error("CAP1 coefficient dimensions disagree")
    output = np.empty_like(values)
    output[0] = values[0]
    for frame in range(1, len(values)):
        prediction = _signed_mod(
            _round_q8(values[frame - 1], model.factors_q8) + model.biases
        )
        output[frame] = _signed_mod(values[frame] - prediction)
    return output


def _restore(residuals: np.ndarray, model: _Ar1Bias) -> np.ndarray:
    values = _require_coefficients(residuals)
    if values.shape[1] != len(model.factors_q8):
        raise Cap1Error("CAP1 residual dimensions disagree")
    output = np.empty_like(values)
    output[0] = values[0]
    for frame in range(1, len(values)):
        prediction = _signed_mod(
            _round_q8(output[frame - 1], model.factors_q8) + model.biases
        )
        output[frame] = _signed_mod(prediction + values[frame])
    return output


def _rice_cost(residuals: np.ndarray) -> int:
    encoded = _zigzag_signed(residuals, 12)
    total = 0
    for dimension in range(encoded.shape[1]):
        column = encoded[:, dimension].astype(np.uint64, copy=False)
        total += min(
            int((column >> candidate).sum()) + column.size * (candidate + 1)
            for candidate in range(12)
        )
    return total


def _fit_dimension(column: np.ndarray) -> tuple[int, int]:
    previous = column[:-1].astype(np.int64)
    target = column[1:].astype(np.int64)
    denominator = int(np.square(previous).sum())
    numerator = 256 * int((previous * target).sum())
    estimated = (
        0 if denominator == 0
        else (1 if numerator >= 0 else -1)
        * ((abs(numerator) + denominator // 2) // denominator)
    )
    centre = min(Q8_MAX, max(Q8_MIN, estimated))
    best: tuple[int, int, int, int] | None = None
    for factor in range(max(Q8_MIN, centre - 96), min(Q8_MAX, centre + 96) + 1):
        baseline = _round_q8(previous, np.full(previous.shape, factor, dtype=np.int16))
        central_bias = min(
            BIAS_MAX,
            max(BIAS_MIN, int(np.median(target - baseline))),
        )
        for bias in range(
            max(BIAS_MIN, central_bias - 4),
            min(BIAS_MAX, central_bias + 4) + 1,
        ):
            residual = _signed_mod(target - _signed_mod(baseline + bias))
            padded = np.concatenate((np.zeros((1, 1), dtype=np.int32), residual[:, None]))
            candidate = (_rice_cost(padded), abs(factor - 256), factor, bias)
            if best is None or candidate < best:
                best = candidate
    assert best is not None
    return best[2], best[3]


def _fit(coefficients: np.ndarray) -> _Ar1Bias:
    values = _require_coefficients(coefficients)
    pairs = [_fit_dimension(values[:, dimension]) for dimension in range(values.shape[1])]
    return _Ar1Bias(
        np.asarray([pair[0] for pair in pairs], dtype=np.int16),
        np.asarray([pair[1] for pair in pairs], dtype=np.int8),
    )


def _pack_model(model: _Ar1Bias) -> bytes:
    return (
        np.asarray(model.factors_q8, dtype="<i2").tobytes()
        + np.asarray(model.biases, dtype="i1").tobytes()
    )


def _unpack_model(raw: bytes, dimensions: int) -> _Ar1Bias:
    if len(raw) != dimensions * 3:
        raise Cap1Error("invalid CAP1 predictor metadata length")
    return _Ar1Bias(
        np.frombuffer(raw[:2 * dimensions], dtype="<i2").copy(),
        np.frombuffer(raw[2 * dimensions:], dtype="i1").copy(),
    )


def _parse_cpr1(raw: bytes, dimensions: int) -> tuple[int, int, dict[str, bytes]]:
    prefix = 12 + 8 * dimensions + 32 + dimensions
    if len(raw) < prefix or raw[:4] != CPR1_MAGIC:
        raise Cap1Error("invalid CPR1 carrier")
    basis_bits, coefficient_bits = struct.unpack_from("<II", raw, 4)
    basis_bytes = (basis_bits + 7) // 8
    coefficient_bytes = (coefficient_bits + 7) // 8
    if not basis_bits or not coefficient_bits or len(raw) != prefix + basis_bytes + coefficient_bytes:
        raise Cap1Error("invalid CPR1 carrier bit counts")
    offset = 12
    fields = {
        "scales": raw[offset:offset + 8 * dimensions],
        "lengths": raw[offset + 8 * dimensions:offset + 8 * dimensions + 32],
        "ks": raw[offset + 8 * dimensions + 32:prefix],
        "basis": raw[prefix:prefix + basis_bytes],
        "coefficients": raw[prefix + basis_bytes:],
    }
    return basis_bits, coefficient_bits, fields


def _coefficients(fields: dict[str, bytes], bits: int, frames: int, dimensions: int) -> np.ndarray:
    encoded = _decode_rice(
        np.frombuffer(fields["ks"], dtype=np.uint8),
        fields["coefficients"],
        bits,
        frames,
        dimensions,
    )
    delta = _unzigzag_unsigned(encoded, 12)
    return _signed_mod(np.cumsum(delta, axis=0, dtype=np.int64))


def encode_cap1(raw_cpr1: bytes, frames: int = 600, dimensions: int = 12) -> bytes:
    """Encode a canonical CPR1 carrier and require exact reconstruction."""
    basis_bits, coefficient_bits, fields = _parse_cpr1(raw_cpr1, dimensions)
    coefficients = _coefficients(fields, coefficient_bits, frames, dimensions)
    model = _fit(coefficients)
    residuals = _residuals(coefficients, model)
    ks, payload, residual_bits = _encode_rice(_zigzag_signed(residuals, 12))
    result = (
        MAGIC
        + bytes((VERSION, 0, 0, 0))
        + _u24(basis_bits)
        + _u24(residual_bits)
        + _pack_model(model)
        + fields["scales"]
        + fields["lengths"]
        + ks.reshape(-1).tobytes()
        + fields["basis"]
        + payload
    )
    if decode_cap1(result, frames, dimensions) != raw_cpr1:
        raise Cap1Error("CAP1 encoder failed CPR1 byte parity")
    return result


def decode_cap1(blob: bytes, frames: int = 600, dimensions: int = 12) -> bytes:
    """Strictly restore the canonical CPR1 carrier bytes."""
    if len(blob) < HEADER_BYTES or blob[:4] != MAGIC:
        raise Cap1Error("invalid CAP1 header")
    if tuple(blob[4:8]) != (VERSION, 0, 0, 0):
        raise Cap1Error("unsupported or noncanonical CAP1 version")
    basis_bits = _read_u24(blob[8:11])
    residual_bits = _read_u24(blob[11:14])
    metadata_bytes = dimensions * 3
    basis_bytes = (basis_bits + 7) // 8
    residual_bytes = (residual_bits + 7) // 8
    expected = HEADER_BYTES + metadata_bytes + 8 * dimensions + 32 + dimensions
    expected += basis_bytes + residual_bytes
    if not basis_bits or not residual_bits or len(blob) != expected:
        raise Cap1Error("invalid CAP1 field lengths")
    offset = HEADER_BYTES
    model = _unpack_model(blob[offset:offset + metadata_bytes], dimensions)
    offset += metadata_bytes
    scales = blob[offset:offset + 8 * dimensions]
    offset += 8 * dimensions
    lengths = blob[offset:offset + 32]
    offset += 32
    ks = np.frombuffer(blob[offset:offset + dimensions], dtype=np.uint8).copy()
    offset += dimensions
    basis = blob[offset:offset + basis_bytes]
    offset += basis_bytes
    residuals = _unzigzag_unsigned(
        _decode_rice(ks, blob[offset:], residual_bits, frames, dimensions),
        12,
    )
    coefficients = _restore(residuals, model)
    previous = np.vstack((np.zeros((1, dimensions), dtype=np.int32), coefficients[:-1]))
    original_delta = _signed_mod(coefficients - previous)
    original_ks, original_payload, original_bits = _encode_rice(
        _zigzag_signed(original_delta, 12)
    )
    result = (
        CPR1_MAGIC
        + struct.pack("<II", basis_bits, original_bits)
        + scales
        + lengths
        + original_ks.reshape(-1).tobytes()
        + basis
        + original_payload
    )
    _parse_cpr1(result, dimensions)
    return result
