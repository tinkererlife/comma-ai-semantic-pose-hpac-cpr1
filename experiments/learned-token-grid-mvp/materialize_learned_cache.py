#!/usr/bin/env python3
"""Replace the semantic maps in a CPR1 target cache with learned hard tokens."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import lzma
from pathlib import Path

import numpy as np
import torch


def load_cache(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".xz":
        with lzma.open(path, "rb") as stream:
            return torch.load(
                io.BytesIO(stream.read()), map_location="cpu", weights_only=False
            )
    return torch.load(path, map_location="cpu", weights_only=False)


def replace_tokens(
    cache: dict[str, torch.Tensor], token_bytes: bytes
) -> tuple[dict[str, torch.Tensor], int]:
    baseline = cache["seg"].to(torch.uint8)
    if len(token_bytes) != baseline.numel():
        raise ValueError(
            f"token byte count {len(token_bytes)} != expected {baseline.numel()}"
        )
    learned = torch.from_numpy(
        np.frombuffer(token_bytes, dtype=np.uint8).copy()
    ).reshape_as(baseline)
    if int(learned.max()) >= 5:
        raise ValueError("learned token IDs must be in [0, 5)")
    changed = int((learned != baseline).sum())
    output = dict(cache)
    output["seg"] = learned
    return output, changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument("--learned-tokens", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    token_bytes = args.learned_tokens.read_bytes()
    output, changed = replace_tokens(load_cache(args.base_cache), token_bytes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.out)
    report = {
        "changed_tokens": changed,
        "token_sha256": hashlib.sha256(token_bytes).hexdigest(),
        "cache_bytes": args.out.stat().st_size,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
