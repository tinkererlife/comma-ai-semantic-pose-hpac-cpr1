"""Lossless RC64 coder and container-marker regression tests."""

from __future__ import annotations

import hashlib
import importlib.util
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from rc64 import NativeDecoder, NativeEncoder, TOTAL, quantize_probabilities  # noqa: E402


def compile_backend(tmp_path: Path) -> Path:
    output = tmp_path / ("rc64.dylib" if sys.platform == "darwin" else "rc64.so")
    shared = "-dynamiclib" if sys.platform == "darwin" else "-shared"
    subprocess.run(
        [
            "cc", "-O3", "-std=c11", shared, "-fPIC",
            str(ROOT / "code/rc64_backend.c"), "-o", str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return output


def test_native_rc64_matches_public_golden_vector(tmp_path):
    generator = np.random.default_rng(130135)
    probabilities = generator.dirichlet(np.ones(5), size=257).astype(np.float32)
    symbols = np.asarray(
        [
            generator.choice(
                5, p=row.astype(np.float64) / row.astype(np.float64).sum()
            )
            for row in probabilities
        ],
        dtype=np.int32,
    )
    frequencies = quantize_probabilities(probabilities)
    assert np.all(frequencies.astype(np.uint64).sum(axis=1) == TOTAL)

    library = compile_backend(tmp_path)
    encoder = NativeEncoder(library)
    encoder.encode(symbols[:113], probabilities[:113])
    encoder.encode(symbols[113:], probabilities[113:])
    stream = encoder.finish()
    assert len(stream) == 59
    assert hashlib.sha256(stream).hexdigest() == (
        "1738f5f620492e59da4f154d34c3566e6c61ad434f9bd5da783ebc50df0c57ea"
    )

    decoder = NativeDecoder(library, stream)
    decoded = np.concatenate([
        decoder.decode(probabilities[:79]),
        decoder.decode(probabilities[79:]),
    ])
    assert np.array_equal(decoded, symbols)


def test_rc64_container_flag_preserves_models_and_accepts_byte_stream(tmp_path):
    module_path = ROOT / "experiments/learned-token-grid-mvp/replace_archive_tokens.py"
    spec = importlib.util.spec_from_file_location("replace_archive_tokens", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    models = b"fixed-model-bytes"
    payload = struct.pack("<I", len(models)) + models + b"\0\0\0\0"
    base = tmp_path / "base.zip"
    with zipfile.ZipFile(base, "w") as archive:
        archive.writestr("p", payload)
    output = tmp_path / "rc64.zip"
    report = module.replace_token_stream(
        base, b"three-byte-stream", output, token_codec="rc64"
    )
    with zipfile.ZipFile(output) as archive:
        rebuilt = archive.read("p")
    field = struct.unpack_from("<I", rebuilt)[0]
    assert field & module.RC64_MODEL_LENGTH_FLAG
    assert field & ~module.RC64_MODEL_LENGTH_FLAG == len(models)
    assert rebuilt[4 : 4 + len(models)] == models
    assert rebuilt[4 + len(models) :] == b"three-byte-stream"
    assert report["token_codec"] == "rc64"

    with pytest.raises(ValueError, match="uint32 words"):
        module.replace_token_stream(
            base, b"odd", tmp_path / "invalid.zip", token_codec="range32"
        )
