"""Cross-check retained checkpoints against the frozen archive lineage."""

from __future__ import annotations

import hashlib
import lzma
import struct
import sys
import zipfile
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))

from hpac_integer import IntegerHPAC  # noqa: E402
from hpac_self_compress import (  # noqa: E402
    enable_self_compression,
    set_deployed_bit_depths,
)
from integer_model_io import deserialize_integer_model  # noqa: E402
from pack_hpac_integer import serialize  # noqa: E402
from pack_hpac_self_compress import (  # noqa: E402
    LZMA_FILTERS,
    model_from_args,
    serialize_self_compressed,
)
from pack_semantic_pose import pack_semantic  # noqa: E402


def base_components() -> tuple[bytes, bytes, bytes]:
    archive_path = ROOT / "artifacts/base/int5_delta_archive.zip"
    with zipfile.ZipFile(archive_path) as archive:
        payload = archive.read("p")
    model_bytes = struct.unpack_from("<I", payload)[0]
    models = lzma.decompress(payload[4 : 4 + model_bytes])
    semantic_bytes, carrier_bytes = struct.unpack_from("<II", models)
    semantic_start = 8
    carrier_start = semantic_start + semantic_bytes
    hpac_start = carrier_start + carrier_bytes
    return (
        models[semantic_start:carrier_start],
        models[hpac_start:],
        payload[4 + model_bytes :],
    )


def test_selected_semantic_checkpoint_is_the_embedded_renderer():
    semantic, _, _ = base_components()
    checkpoint = torch.load(
        ROOT
        / "artifacts/checkpoints/"
        "semantic_renderer_w96_b4_qat4_fixedtau05_tail6k_lr2e7.pt",
        map_location="cpu",
        weights_only=False,
    )
    packed, _ = pack_semantic(checkpoint)
    assert packed == semantic
    assert hashlib.sha256(packed).hexdigest() == (
        "9b98360bd56918b5a414ace375c29790b7fe9f7f55cf423c0564ef4e62a39b99"
    )


def test_base_hpac_can_be_losslessly_extracted_and_reserialized():
    _, expected, _ = base_components()
    model = IntegerHPAC(
        channels=64,
        patch=64,
        delta=2,
        frame_dim=8,
        norm_mode="none",
        activation="relu",
        use_frame_scale=True,
        weight_bound=127,
        activation_bound=127,
        use_weight_scales=True,
        weight_exponent_min=-6,
        use_spm=True,
        use_norm_gates=False,
    )
    deserialize_integer_model(model, expected)
    assert serialize(model) == expected


def test_selected_hpac_checkpoint_reproduces_the_packed_model():
    class Args:
        channels = 64
        patch = 64
        delta = 2
        frame_dim = 8
        weight_bound = 127
        activation_bound = 127
        weight_exponent_min = -6

    checkpoint = torch.load(
        ROOT / "artifacts/checkpoints/hpac_selfcompress_l1_fastbits_e60.pt",
        map_location="cpu",
        weights_only=False,
    )
    model = model_from_args(Args(), True)
    model.load_state_dict(checkpoint["state_dict"])
    set_deployed_bit_depths(model, True)
    raw = serialize_self_compressed(model)
    actual = lzma.compress(raw, format=lzma.FORMAT_XZ, filters=LZMA_FILTERS)
    expected = (
        ROOT / "artifacts/hpac/hpac_selfcompress_l1_fastbits_e60.bin.xz"
    ).read_bytes()
    assert actual == expected
    assert len(raw) == 20_179


def test_self_compressed_weights_respect_runtime_weight_bound():
    model = IntegerHPAC(channels=4, patch=64, weight_bound=127)
    enable_self_compression(model, init_bits=8.0)
    set_deployed_bit_depths(model, True)
    with torch.no_grad():
        model.conv_a.weight[0, 0, 2, 3] = -128
    assert model.conv_a.codes()[0][0, 0, 2, 3].item() == -127


def test_final_token_stream_is_the_rebuilt_stream():
    _, _, old_tokens = base_components()
    selected = (
        ROOT / "artifacts/hpac/hpac_selfcompress_l1_fastbits_e60.tokens.bin"
    ).read_bytes()
    assert old_tokens != selected
    assert len(selected) == 116_980
    assert hashlib.sha256(selected).hexdigest() == (
        "948379872ff81a4e5d948ec301c143be00ebd0033544c8abdfb4af0f4c4a15eb"
    )
