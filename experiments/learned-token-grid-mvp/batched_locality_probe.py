#!/usr/bin/env python3
"""Compare batched localized HPAC deltas with the exact serial path."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from hpac_token_search import HPACRateOracle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe-root", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = parse_args()
    if not 1 <= args.samples <= 299:
        raise ValueError("samples must lie in [1, 299]")
    device = torch.device(args.device)
    recipe_root = args.recipe_root.resolve()
    sys.path.insert(0, str(recipe_root / "code"))
    raw = np.frombuffer(args.tokens.read_bytes(), dtype=np.uint8).copy()
    tokens = torch.from_numpy(raw).reshape(600, 384, 512)
    oracle = HPACRateOracle(
        recipe_root
        / "artifacts/checkpoints/hpac_selfcompress_l1_fastbits_e60.pt",
        recipe_root / "code",
        tokens,
        0,
        device,
    )
    generator = random.Random(args.seed)
    frames = generator.sample(range(0, 598, 2), args.samples)
    candidates = []
    boundary_positions = [
        (0, 0), (63, 63), (64, 64), (127, 255), (383, 511)
    ]
    for sample, frame in enumerate(frames):
        if sample < len(boundary_positions):
            row, col = boundary_positions[sample]
        else:
            row = generator.randrange(384)
            col = generator.randrange(512)
        candidate = tokens[frame].clone()
        before = int(candidate[row, col])
        candidate[row, col] = (before + generator.randrange(1, 5)) % 5
        candidates.append(candidate)

    # Warm the CUDA kernels without including first-use overhead in either path.
    oracle.localized_move_delta(tokens, frames[0], candidates[0])
    oracle.localized_move_delta_batch(tokens, frames[:1], torch.stack(candidates[:1]))
    synchronize(device)

    started = time.perf_counter()
    serial = [
        oracle.localized_move_delta(tokens, frame, candidate)
        for frame, candidate in zip(frames, candidates)
    ]
    synchronize(device)
    serial_seconds = time.perf_counter() - started

    started = time.perf_counter()
    batched = oracle.localized_move_delta_batch(
        tokens, frames, torch.stack(candidates)
    )
    synchronize(device)
    batch_seconds = time.perf_counter() - started

    records = []
    for frame, old, new in zip(frames, serial, batched):
        serial_delta, serial_frames, serial_locality = old
        batch_delta, batch_frames, batch_locality = new
        frame_errors = {
            str(index): batch_frames[index] - value
            for index, value in serial_frames.items()
        }
        records.append({
            "frame": frame,
            "serial_delta_bits": serial_delta,
            "batch_delta_bits": batch_delta,
            "delta_error_bits": batch_delta - serial_delta,
            "frame_delta_errors_bits": frame_errors,
            "locality_equal": batch_locality == serial_locality,
        })
    max_error = max(
        abs(error)
        for record in records
        for error in [
            record["delta_error_bits"],
            *record["frame_delta_errors_bits"].values(),
        ]
    )
    report = {
        "samples": args.samples,
        "serial_seconds": serial_seconds,
        "batch_seconds": batch_seconds,
        "speedup": serial_seconds / batch_seconds,
        "max_abs_error_bits": max_error,
        "all_locality_equal": all(record["locality_equal"] for record in records),
        "all_exact": max_error < 1e-9,
        "records": records,
    }
    print(json.dumps(report, indent=2), flush=True)
    if not report["all_exact"] or not report["all_locality_equal"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
