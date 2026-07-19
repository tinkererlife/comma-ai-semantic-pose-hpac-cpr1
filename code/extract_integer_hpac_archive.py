#!/usr/bin/env python3
"""Extract the deployed integer HPAC state from a submission archive."""

from __future__ import annotations

import argparse
import json
import lzma
import zipfile
from pathlib import Path

import numpy as np
import torch

from hpac_integer import IntegerHPAC


def read_payload(path: Path) -> bytes:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            return archive.read("p")
    return path.read_bytes()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--patch", type=int, default=64)
    parser.add_argument("--delta", type=int, default=2)
    parser.add_argument("--frame-dim", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from integer_model_io import deserialize_integer_model

    payload = read_payload(args.archive)
    models_bytes = int(np.frombuffer(payload[:4], dtype=np.uint32)[0])
    models_raw = lzma.decompress(payload[4:4 + models_bytes])
    semantic_bytes, carrier_bytes = np.frombuffer(models_raw[:8], dtype=np.uint32)
    semantic_pose_bytes = 8 + int(semantic_bytes) + int(carrier_bytes)
    hpac_raw = models_raw[semantic_pose_bytes:]

    model = IntegerHPAC(
        channels=args.channels,
        patch=args.patch,
        delta=args.delta,
        frame_dim=args.frame_dim,
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
    deserialize_integer_model(model, hpac_raw)
    state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    config = {
        "source_archive": str(args.archive),
        "channels": args.channels,
        "patch": args.patch,
        "delta": args.delta,
        "frame_dim": args.frame_dim,
        "norm_mode": "none",
        "activation": "relu",
        "frame_scale": True,
        "weight_bound": 127,
        "activation_bound": 127,
        "weight_scales": True,
        "weight_exponent_min": -6,
        "spm": True,
        "norm_gates": False,
        "target_mode": "raw",
        "deployed_bytes": len(hpac_raw),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state, "config": config}, args.out)
    print(json.dumps({
        "output": str(args.out),
        "deployed_bytes": len(hpac_raw),
        "state_tensors": len(state),
        "state_values": sum(value.numel() for value in state.values()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
