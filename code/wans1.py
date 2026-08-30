"""Exact WANS1 representation for the fixed CPR1 semantic renderer.

Adapted from codexblack's public PR #135 ExperimentBook implementation.  This
module deliberately exposes a byte-to-byte API: the decoded output is the
legacy 40,252-byte CPR1 renderer payload consumed by ``inflate.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
import struct

import numpy as np


MAGIC = b"WANS"
VERSION = 1
PRECISION = 12
TOTAL = 1 << PRECISION
RANS_L = 1 << 23
STATE_BYTES = 4
ALPHABET = 16
RESCALE_AT = TOTAL * 2


class Wans1Error(ValueError):
    """A renderer payload or adaptive-rANS stream is invalid."""


@dataclass(frozen=True)
class _Schema:
    name: str
    shape: tuple[int, ...]

    @property
    def count(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))

    @property
    def fp16(self) -> bool:
        return len(self.shape) < 2

    @property
    def scale_count(self) -> int:
        if self.fp16:
            return 0
        return self.shape[-1] if self.name.endswith("embed.weight") else self.shape[0]


def _schema() -> tuple[_Schema, ...]:
    items: list[tuple[str, tuple[int, ...]]] = [
        ("token_embed.weight", (5, 96)),
        ("frame_embed.weight", (600, 8)),
        ("coord_mix.weight", (96, 100, 1, 1)),
        ("coord_mix.bias", (96,)),
    ]
    for block in range(4):
        prefix = f"blocks.{block}"
        items.extend([
            (f"{prefix}.dw.weight", (96, 1, 3, 3)),
            (f"{prefix}.dw.bias", (96,)),
            (f"{prefix}.pw.weight", (96, 96, 1, 1)),
            (f"{prefix}.pw.bias", (96,)),
            (f"{prefix}.norm.weight", (96,)),
            (f"{prefix}.norm.bias", (96,)),
            (f"{prefix}.film.weight", (192, 8)),
            (f"{prefix}.film.bias", (192,)),
        ])
    items.extend([("head.weight", (3, 96, 3, 3)), ("head.bias", (3,))])
    return tuple(_Schema(name, shape) for name, shape in items)


SCHEMA = _schema()
W4_COUNT = sum(not item.fp16 for item in SCHEMA)
MASK_BYTES = (W4_COUNT + 7) // 8
PRIOR_BYTES = (W4_COUNT * 2 + 7) // 8
OFFSET_BYTES = 2 * (W4_COUNT - 1)
HEADER_BYTES = 5 + MASK_BYTES + PRIOR_BYTES + OFFSET_BYTES


def _prior(index: int) -> np.ndarray:
    if not 0 <= index <= 3:
        raise Wans1Error("unsupported WANS1 prior")
    centre = ALPHABET // 2
    distance = np.abs(np.arange(ALPHABET, dtype=np.int64) - centre)
    return (1 + (4 + 2 * index) * np.maximum(0, centre - distance)).astype(np.uint32)


def _normalise(counts: np.ndarray) -> np.ndarray:
    source = np.asarray(counts, dtype=np.uint64)
    if source.shape != (ALPHABET,) or np.any(source == 0):
        raise Wans1Error("invalid WANS1 count vector")
    total = int(source.sum())
    product = source * TOTAL
    frequencies = np.maximum(1, product // total).astype(np.int64)
    remainder = int(TOTAL - frequencies.sum())
    fractions = product % total
    if remainder > 0:
        for index in np.argsort(-fractions, kind="stable")[:remainder]:
            frequencies[index] += 1
    elif remainder < 0:
        candidates = [
            int(index) for index in np.argsort(fractions, kind="stable")
            if frequencies[index] > 1
        ]
        if len(candidates) < -remainder:
            raise Wans1Error("WANS1 normalization underflow")
        for index in candidates[:-remainder]:
            frequencies[index] -= 1
    if int(frequencies.sum()) != TOTAL or np.any(frequencies <= 0):
        raise Wans1Error("WANS1 normalization failed")
    return frequencies.astype(np.uint16)


def _update(counts: np.ndarray, symbol: int) -> None:
    counts[symbol] += 1
    if int(counts.sum(dtype=np.uint64)) >= RESCALE_AT:
        counts[:] = np.maximum(1, (counts + 1) // 2)


def _encode_adaptive(symbols: np.ndarray, prior_index: int) -> bytes:
    values = np.asarray(symbols, dtype=np.int32).reshape(-1)
    if not values.size or np.any(values < 0) or np.any(values >= ALPHABET):
        raise Wans1Error("WANS1 symbol outside the fixed alphabet")
    counts = _prior(prior_index)
    events = np.empty((values.size, ALPHABET), dtype=np.uint16)
    for index, symbol in enumerate(values):
        events[index] = _normalise(counts)
        _update(counts, int(symbol))
    state = RANS_L
    emitted = bytearray()
    for symbol, frequency in zip(values[::-1], events[::-1], strict=True):
        symbol = int(symbol)
        freq = int(frequency[symbol])
        cumulative = int(frequency[:symbol].sum(dtype=np.uint32))
        maximum = ((RANS_L >> PRECISION) << 8) * freq
        while state >= maximum:
            emitted.append(state & 0xFF)
            state >>= 8
        state = ((state // freq) << PRECISION) + (state % freq) + cumulative
    if not RANS_L <= state < 1 << 32:
        raise Wans1Error("WANS1 state does not fit its header")
    return state.to_bytes(STATE_BYTES, "little") + bytes(reversed(emitted))


def _decode_adaptive(payload: bytes, count: int, prior_index: int) -> np.ndarray:
    if count <= 0 or len(payload) < STATE_BYTES:
        raise Wans1Error("truncated WANS1 state")
    state = int.from_bytes(payload[:STATE_BYTES], "little")
    if state < RANS_L:
        raise Wans1Error("invalid WANS1 initial state")
    cursor = STATE_BYTES
    output = np.empty(count, dtype=np.uint8)
    counts = _prior(prior_index)
    for index in range(count):
        frequencies = _normalise(counts)
        slot = state & (TOTAL - 1)
        cumulative = np.cumsum(frequencies, dtype=np.uint32)
        symbol = int(np.searchsorted(cumulative, slot, side="right"))
        if symbol >= ALPHABET:
            raise Wans1Error("WANS1 state selects no symbol")
        lower = 0 if symbol == 0 else int(cumulative[symbol - 1])
        state = int(frequencies[symbol]) * (state >> PRECISION) + slot - lower
        while state < RANS_L:
            if cursor >= len(payload):
                raise Wans1Error("truncated WANS1 renormalization bytes")
            state = (state << 8) | payload[cursor]
            cursor += 1
        output[index] = symbol
        _update(counts, symbol)
    if cursor != len(payload):
        raise Wans1Error("WANS1 trailing bytes")
    return output


def _unpack_w4(blob: bytes, count: int) -> np.ndarray:
    if len(blob) != (count + 1) // 2:
        raise Wans1Error("invalid WANS1 raw W4 stream length")
    packed = np.frombuffer(blob, dtype=np.uint8)
    values = np.empty(packed.size * 2, dtype=np.int8)
    values[0::2] = packed & 0xF
    values[1::2] = packed >> 4
    values[values >= 8] -= 16
    if count & 1 and packed[-1] >> 4:
        raise Wans1Error("nonzero WANS1 nibble padding")
    values = values[:count]
    if np.any(values == -8):
        raise Wans1Error("reserved WANS1 -8 code")
    return values


def _pack_w4(codes: np.ndarray) -> bytes:
    values = np.asarray(codes, dtype=np.int16).reshape(-1)
    if not values.size or np.any(values < -7) or np.any(values > 7):
        raise Wans1Error("WANS1 code outside CPR1 int4 range")
    unsigned = (values & 0xF).astype(np.uint8)
    if unsigned.size & 1:
        unsigned = np.pad(unsigned, (0, 1))
    return (unsigned[0::2] | (unsigned[1::2] << 4)).tobytes()


def _parse_legacy(blob: bytes) -> tuple[bytes, list[bytes], list[np.ndarray]]:
    offset = 0
    metadata = bytearray()
    raw_streams: list[bytes] = []
    symbols: list[np.ndarray] = []
    for item in SCHEMA:
        if item.fp16:
            size = item.count * 2
            raw = blob[offset:offset + size]
            if len(raw) != size:
                raise Wans1Error(f"truncated semantic tensor {item.name}")
            metadata.extend(raw)
            offset += size
            continue
        scale_bytes = item.scale_count * 2
        scales = blob[offset:offset + scale_bytes]
        offset += scale_bytes
        code_bytes = (item.count + 1) // 2
        raw = blob[offset:offset + code_bytes]
        offset += code_bytes
        if len(scales) != scale_bytes or len(raw) != code_bytes:
            raise Wans1Error(f"truncated semantic tensor {item.name}")
        codes = _unpack_w4(raw, item.count)
        metadata.extend(scales)
        raw_streams.append(raw)
        symbols.append((codes.astype(np.int16) + 8).astype(np.uint8))
    if offset != len(blob):
        raise Wans1Error("legacy semantic payload has trailing bytes")
    return bytes(metadata), raw_streams, symbols


def _pack_priors(priors: list[int]) -> bytes:
    if len(priors) != W4_COUNT or any(not 0 <= value <= 3 for value in priors):
        raise Wans1Error("invalid WANS1 prior catalog")
    output = bytearray(PRIOR_BYTES)
    for index, value in enumerate(priors):
        output[index // 4] |= value << (2 * (index % 4))
    return bytes(output)


def _unpack_priors(raw: bytes) -> list[int]:
    if len(raw) != PRIOR_BYTES:
        raise Wans1Error("truncated WANS1 prior table")
    return [(raw[index // 4] >> (2 * (index % 4))) & 3 for index in range(W4_COUNT)]


def encode_wans1(legacy: bytes) -> bytes:
    """Encode one exact legacy CPR1 renderer payload."""
    metadata, raw_streams, symbols = _parse_legacy(legacy)
    trials = [[_encode_adaptive(stream, prior) for prior in range(4)] for stream in symbols]
    selected = [min(range(4), key=lambda prior: len(trial[prior])) for trial in trials]
    modes: list[bool] = []
    priors: list[int] = []
    streams: list[bytes] = []
    for raw, trial, prior in zip(raw_streams, trials, selected, strict=True):
        enabled = len(trial[prior]) < len(raw)
        modes.append(enabled)
        priors.append(prior if enabled else 0)
        streams.append(trial[prior] if enabled else raw)
    mask = bytearray(MASK_BYTES)
    for index, enabled in enumerate(modes):
        if enabled:
            mask[index // 8] |= 1 << (index % 8)
    ends = np.cumsum([len(stream) for stream in streams], dtype=np.int64)
    if int(ends[-1]) >= 1 << 16:
        raise Wans1Error("WANS1 stream area exceeds u16 offsets")
    offsets = b"".join(struct.pack("<H", int(end)) for end in ends[:-1])
    result = MAGIC + bytes((VERSION,)) + bytes(mask) + _pack_priors(priors)
    result += offsets + metadata + b"".join(streams)
    if decode_wans1(result) != legacy:
        raise Wans1Error("WANS1 encoder failed byte parity")
    return result


def decode_wans1(blob: bytes) -> bytes:
    """Restore the exact legacy 40,252-byte semantic renderer payload."""
    if len(blob) < HEADER_BYTES or blob[:4] != MAGIC or blob[4] != VERSION:
        raise Wans1Error("invalid WANS1 header")
    offset = 5
    mask = blob[offset:offset + MASK_BYTES]
    offset += MASK_BYTES
    if W4_COUNT % 8 and mask[-1] & ~((1 << (W4_COUNT % 8)) - 1):
        raise Wans1Error("nonzero WANS1 mask padding")
    priors = _unpack_priors(blob[offset:offset + PRIOR_BYTES])
    offset += PRIOR_BYTES
    ends = [
        struct.unpack_from("<H", blob, offset + 2 * index)[0]
        for index in range(W4_COUNT - 1)
    ]
    offset += OFFSET_BYTES
    metadata_bytes = sum(
        item.count * 2 if item.fp16 else item.scale_count * 2 for item in SCHEMA
    )
    if len(blob) < offset + metadata_bytes + W4_COUNT:
        raise Wans1Error("truncated WANS1 payload")
    metadata = blob[offset:offset + metadata_bytes]
    streams = blob[offset + metadata_bytes:]
    ends.append(len(streams))
    starts = [0] + ends[:-1]
    if any(end <= start for start, end in zip(starts, ends, strict=True)):
        raise Wans1Error("non-monotonic WANS1 stream offsets")
    modes = [bool(mask[index // 8] & (1 << (index % 8))) for index in range(W4_COUNT)]
    if any(prior and not mode for prior, mode in zip(priors, modes, strict=True)):
        raise Wans1Error("raw WANS1 stream has a noncanonical prior")
    result = bytearray()
    metadata_offset = 0
    stream_index = 0
    for item in SCHEMA:
        if item.fp16:
            size = item.count * 2
            result.extend(metadata[metadata_offset:metadata_offset + size])
            metadata_offset += size
            continue
        scale_bytes = item.scale_count * 2
        result.extend(metadata[metadata_offset:metadata_offset + scale_bytes])
        metadata_offset += scale_bytes
        stream = streams[starts[stream_index]:ends[stream_index]]
        codes = (
            _decode_adaptive(stream, item.count, priors[stream_index]).astype(np.int16) - 8
            if modes[stream_index]
            else _unpack_w4(stream, item.count)
        )
        result.extend(_pack_w4(codes))
        stream_index += 1
    if metadata_offset != len(metadata) or stream_index != W4_COUNT:
        raise Wans1Error("WANS1 field accounting mismatch")
    return bytes(result)
