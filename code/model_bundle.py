"""Versioned lossless representations inside the CPR1 model bundle."""

from __future__ import annotations

from dataclasses import dataclass
import lzma
import struct


RC64_MODEL_LENGTH_FLAG = 1 << 31
WANS1_MODEL_LENGTH_FLAG = 1 << 30
CAP1_MODEL_LENGTH_FLAG = 1 << 29
MODEL_LENGTH_MASK = (1 << 29) - 1
KNOWN_MODEL_FLAGS = (
    RC64_MODEL_LENGTH_FLAG | WANS1_MODEL_LENGTH_FLAG | CAP1_MODEL_LENGTH_FLAG
)

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


@dataclass(frozen=True)
class ModelBundle:
    semantic: bytes
    carrier: bytes
    hpac: bytes
    token_codec: str
    semantic_codec: str
    carrier_codec: str


def parse_model_field(field: int) -> tuple[int, int]:
    """Return compressed byte length and known feature flags."""
    length = field & MODEL_LENGTH_MASK
    flags = field & ~MODEL_LENGTH_MASK
    if flags & ~KNOWN_MODEL_FLAGS:
        raise ValueError(f"unknown CPR1 model flags: 0x{flags:08x}")
    return length, flags


def decode_model_bundle(compressed: bytes, flags: int) -> ModelBundle:
    """Decode all storage-only layers to the canonical legacy model bytes."""
    raw = lzma.decompress(compressed)
    if len(raw) < 8:
        raise ValueError("model bundle is truncated before semantic-pose lengths")
    semantic_bytes, carrier_bytes = struct.unpack_from("<II", raw)
    stored_end = 8 + semantic_bytes + carrier_bytes
    if stored_end > len(raw):
        raise ValueError("semantic-pose lengths exceed the model bundle")
    semantic = raw[8:8 + semantic_bytes]
    carrier = raw[8 + semantic_bytes:stored_end]
    if flags & WANS1_MODEL_LENGTH_FLAG:
        from wans1 import decode_wans1

        semantic = decode_wans1(semantic)
    if flags & CAP1_MODEL_LENGTH_FLAG:
        from cap1 import decode_cap1

        carrier = decode_cap1(carrier)
    return ModelBundle(
        semantic=semantic,
        carrier=carrier,
        hpac=raw[stored_end:],
        token_codec="rc64" if flags & RC64_MODEL_LENGTH_FLAG else "range32",
        semantic_codec="wans1" if flags & WANS1_MODEL_LENGTH_FLAG else "legacy",
        carrier_codec="cap1" if flags & CAP1_MODEL_LENGTH_FLAG else "legacy",
    )


def encode_model_bundle(
    bundle: ModelBundle,
    *,
    semantic_codec: str,
    carrier_codec: str,
) -> tuple[bytes, int]:
    """Encode canonical bytes and return compressed bundle plus feature flags."""
    semantic = bundle.semantic
    carrier = bundle.carrier
    flags = RC64_MODEL_LENGTH_FLAG if bundle.token_codec == "rc64" else 0
    if semantic_codec == "wans1":
        from wans1 import encode_wans1

        semantic = encode_wans1(semantic)
        flags |= WANS1_MODEL_LENGTH_FLAG
    elif semantic_codec != "legacy":
        raise ValueError(f"unsupported semantic codec: {semantic_codec}")
    if carrier_codec == "cap1":
        from cap1 import encode_cap1

        carrier = encode_cap1(carrier)
        flags |= CAP1_MODEL_LENGTH_FLAG
    elif carrier_codec != "legacy":
        raise ValueError(f"unsupported carrier codec: {carrier_codec}")
    raw = (
        struct.pack("<II", len(semantic), len(carrier))
        + semantic
        + carrier
        + bundle.hpac
    )
    compressed = lzma.compress(raw, format=lzma.FORMAT_XZ, filters=LZMA_FILTERS)
    if len(compressed) > MODEL_LENGTH_MASK:
        raise ValueError("compressed model bundle exceeds the CPR1 length field")
    return compressed, flags
