"""Native 63-bit arithmetic coder for five-class HPAC probabilities.

This is the lossless RC64 representation described by challenge PR #135 and
adapted from its public ExperimentBook implementation; see the lineage file.
It maps each canonical float32 probability row to five positive integer
frequencies totalling 2**31, then streams them through one 63-bit interval.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np


ALPHABET = 5
TOTAL = 1 << 31


class Rc64Error(ValueError):
    """The RC64 library, probability rows, or stream are invalid."""


def quantize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Convert canonical float32 rows to deterministic positive u31 counts."""
    values = np.asarray(probabilities, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != ALPHABET or not values.size:
        raise Rc64Error("RC64 probabilities must have shape [N, 5]")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise Rc64Error("RC64 probabilities must be finite and positive")
    if np.any(np.abs(values.astype(np.float64).sum(axis=1) - 1.0) > 2e-5):
        raise Rc64Error("RC64 probability rows must sum to one")
    frequencies = np.floor(values.astype(np.float64) * TOTAL).astype(np.int64)
    np.maximum(frequencies, 1, out=frequencies)
    winners = values.argmax(axis=1)
    frequencies[np.arange(len(values)), winners] += TOTAL - frequencies.sum(axis=1)
    if np.any(frequencies <= 0) or np.any(frequencies >= TOTAL):
        raise Rc64Error("RC64 frequency normalization failed")
    if np.any(frequencies.sum(axis=1) != TOTAL):
        raise Rc64Error("RC64 frequencies do not total 2**31")
    return np.ascontiguousarray(frequencies, dtype=np.uint32)


def _load_library(path: Path):
    library = ctypes.CDLL(str(path.resolve()))
    u8 = ctypes.POINTER(ctypes.c_uint8)
    u32 = ctypes.POINTER(ctypes.c_uint32)
    i32 = ctypes.POINTER(ctypes.c_int32)
    f32 = ctypes.POINTER(ctypes.c_float)
    library.rc64_encoder_create.restype = ctypes.c_void_p
    library.rc64_encoder_destroy.argtypes = [ctypes.c_void_p]
    library.rc64_encoder_encode.argtypes = [
        ctypes.c_void_p, i32, u32, ctypes.c_size_t
    ]
    library.rc64_encoder_encode.restype = ctypes.c_int
    library.rc64_encoder_finish.argtypes = [ctypes.c_void_p]
    library.rc64_encoder_finish.restype = ctypes.c_int
    library.rc64_encoder_data.argtypes = [ctypes.c_void_p]
    library.rc64_encoder_data.restype = u8
    library.rc64_encoder_size.argtypes = [ctypes.c_void_p]
    library.rc64_encoder_size.restype = ctypes.c_size_t
    library.rc64_decoder_create.argtypes = [u8, ctypes.c_size_t]
    library.rc64_decoder_create.restype = ctypes.c_void_p
    library.rc64_decoder_destroy.argtypes = [ctypes.c_void_p]
    library.rc64_decoder_decode_probabilities.argtypes = [
        ctypes.c_void_p, f32, ctypes.c_size_t, i32
    ]
    library.rc64_decoder_decode_probabilities.restype = ctypes.c_int
    library.rc64_decoder_bit_position.argtypes = [ctypes.c_void_p]
    library.rc64_decoder_bit_position.restype = ctypes.c_size_t
    library.rc64_total_frequency.restype = ctypes.c_uint64
    if library.rc64_total_frequency() != TOTAL:
        raise Rc64Error("RC64 library uses an incompatible frequency total")
    return library


class NativeEncoder:
    """Streaming RC64 encoder backed by ``rc64_backend.c``."""

    def __init__(self, library_path: Path) -> None:
        self.library = _load_library(library_path)
        self.context = self.library.rc64_encoder_create()
        if not self.context:
            raise Rc64Error("RC64 encoder allocation failed")
        self.finished = False

    def close(self) -> None:
        if self.context:
            self.library.rc64_encoder_destroy(self.context)
            self.context = None

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown varies.
        self.close()

    def encode(self, symbols: np.ndarray, probabilities: np.ndarray) -> None:
        source = np.ascontiguousarray(symbols, dtype=np.int32).reshape(-1)
        frequencies = quantize_probabilities(probabilities)
        if frequencies.shape[0] != source.size:
            raise Rc64Error("RC64 symbol/probability shapes disagree")
        status = self.library.rc64_encoder_encode(
            self.context,
            source.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            frequencies.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            source.size,
        )
        if status:
            raise Rc64Error(f"native RC64 encode failed with status {status}")

    def finish(self) -> bytes:
        if not self.finished:
            status = self.library.rc64_encoder_finish(self.context)
            if status:
                raise Rc64Error(f"native RC64 finish failed with status {status}")
            self.finished = True
        size = self.library.rc64_encoder_size(self.context)
        pointer = self.library.rc64_encoder_data(self.context)
        if not size or not pointer:
            raise Rc64Error("native RC64 encoder returned an empty stream")
        return ctypes.string_at(pointer, size)


class NativeDecoder:
    """Streaming RC64 decoder using the same canonical float32 rows."""

    def __init__(self, library_path: Path, payload: bytes) -> None:
        if not payload:
            raise Rc64Error("RC64 stream is empty")
        self.library = _load_library(library_path)
        self.payload = (ctypes.c_uint8 * len(payload)).from_buffer_copy(payload)
        self.context = self.library.rc64_decoder_create(self.payload, len(payload))
        if not self.context:
            raise Rc64Error("RC64 decoder allocation failed")

    def close(self) -> None:
        if self.context:
            self.library.rc64_decoder_destroy(self.context)
            self.context = None

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown varies.
        self.close()

    def decode(self, probabilities: np.ndarray) -> np.ndarray:
        values = np.ascontiguousarray(probabilities, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != ALPHABET or not values.size:
            raise Rc64Error("RC64 probabilities must have shape [N, 5]")
        output = np.empty(values.shape[0], dtype=np.int32)
        status = self.library.rc64_decoder_decode_probabilities(
            self.context,
            values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            len(values),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        )
        if status:
            raise Rc64Error(f"native RC64 decode failed with status {status}")
        return output

    @property
    def bit_position(self) -> int:
        return int(self.library.rc64_decoder_bit_position(self.context))
