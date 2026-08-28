#!/usr/bin/env python3
"""Empirically prove which single-token effects are safe to recompute locally."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from hpac_token_search import HPACRateOracle
from learned_token_mvp import (
    N_TOKENS,
    add_recipe_imports,
    camera_and_seg_input,
    load_deployed_submission,
    load_xz_torch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe-root", type=Path, required=True)
    parser.add_argument("--challenge-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def support(value: torch.Tensor, tolerance: float = 0.0) -> dict[str, object]:
    if value.ndim > 2:
        value = value.reshape(-1, *value.shape[-2:]).abs().amax(dim=0)
    else:
        value = value.abs()
    changed = (value > tolerance).nonzero(as_tuple=False)
    if not len(changed):
        return {"count": 0, "bbox": None, "max_abs": float(value.max())}
    low = changed.amin(dim=0).tolist()
    high = changed.amax(dim=0).tolist()
    return {
        "count": len(changed),
        "bbox": [*low, *high],
        "max_abs": float(value.max()),
    }


def main() -> None:
    args = parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be positive")
    device = torch.device(args.device)
    recipe_root = args.recipe_root.resolve()
    challenge_root = args.challenge_root.resolve()
    add_recipe_imports(recipe_root, challenge_root)

    import modules  # pylint: disable=import-error,import-outside-toplevel

    cache = load_xz_torch(
        recipe_root / "artifacts/caches/gt_cache_600_official_ada.pt.xz"
    )
    shape = cache["seg"].shape
    raw = np.frombuffer(args.tokens.read_bytes(), dtype=np.uint8).copy()
    tokens = torch.from_numpy(raw).reshape(shape)
    model, _, _, _ = load_deployed_submission(args.archive, device)
    model.eval()
    segnet = modules.SegNet().eval().to(device)
    segnet.load_state_dict(load_file(modules.segnet_sd_path, device=str(device)))
    hpac = HPACRateOracle(
        recipe_root
        / "artifacts/checkpoints/hpac_selfcompress_l1_fastbits_e60.pt",
        recipe_root / "code",
        tokens,
        0,
        device,
    )

    generator = random.Random(args.seed)
    boundary_cases = [
        (0, 0),
        (63, 63),
        (64, 64),
        (127, 255),
        (383, 511),
    ]
    records = []
    for sample in range(args.samples):
        frame = generator.randrange(0, len(tokens) - 1)
        if sample < len(boundary_cases):
            row, col = boundary_cases[sample]
            position_source = "boundary"
        else:
            row = generator.randrange(tokens.shape[1])
            col = generator.randrange(tokens.shape[2])
            position_source = "random"
        before_id = int(tokens[frame, row, col])
        after_id = (before_id + generator.randrange(1, N_TOKENS)) % N_TOKENS
        candidate = tokens[frame].clone()
        candidate[row, col] = after_id
        previous = (
            torch.zeros_like(tokens[0]) if frame == 0 else tokens[frame - 1]
        )

        started = time.perf_counter()
        before_current, before_current_costs = hpac.frame_bits(
            frame, tokens[frame], previous, return_costs=True
        )
        after_current, after_current_costs = hpac.frame_bits(
            frame, candidate, previous, return_costs=True
        )
        before_next, before_next_costs = hpac.frame_bits(
            frame + 1, tokens[frame + 1], tokens[frame], return_costs=True
        )
        after_next, after_next_costs = hpac.frame_bits(
            frame + 1, tokens[frame + 1], candidate, return_costs=True
        )
        full_seconds = time.perf_counter() - started
        full_delta = (
            after_current + after_next - before_current - before_next
        )

        started = time.perf_counter()
        local_delta, _, locality = hpac.localized_move_delta(
            tokens, frame, candidate
        )
        local_seconds = time.perf_counter() - started

        pair_id = torch.tensor([frame], dtype=torch.long, device=device)
        with torch.no_grad():
            before_eval = model(tokens[frame : frame + 1].long().to(device), pair_id)
            after_eval = model(candidate[None].long().to(device), pair_id)
            before_camera, before_seg_input = camera_and_seg_input(before_eval)
            after_camera, after_seg_input = camera_and_seg_input(after_eval)
            before_seg = segnet(before_seg_input)
            after_seg = segnet(after_seg_input)

        renderer_float = support(after_eval - before_eval)
        renderer_camera = support(after_camera - before_camera)
        seg_logits = support(after_seg - before_seg)
        seg_labels = support(
            after_seg.argmax(dim=1).float() - before_seg.argmax(dim=1).float()
        )
        current_cost_support = support(
            (after_current_costs - before_current_costs).movedim(-1, 0)
        )
        next_cost_support = support(
            (after_next_costs - before_next_costs).movedim(-1, 0)
        )
        record = {
            "sample": sample,
            "frame": frame,
            "position_source": position_source,
            "row": row,
            "col": col,
            "before": before_id,
            "after": after_id,
            "hpac_full_delta_bits": full_delta,
            "hpac_local_delta_bits": local_delta,
            "hpac_delta_error_bits": local_delta - full_delta,
            "hpac_full_seconds": full_seconds,
            "hpac_local_seconds": local_seconds,
            "hpac_speedup": full_seconds / local_seconds,
            "hpac_locality": locality,
            "hpac_current_cost_support": current_cost_support,
            "hpac_next_cost_support": next_cost_support,
            "renderer_has_global_groupnorm": any(
                isinstance(module, torch.nn.GroupNorm) for module in model.modules()
            ),
            "renderer_float_support": renderer_float,
            "renderer_rounded_camera_support": renderer_camera,
            "segnet_logit_support": seg_logits,
            "segnet_label_support": seg_labels,
        }
        records.append(record)
        print(json.dumps(record), flush=True)

    report = {
        "samples": args.samples,
        "all_hpac_deltas_exact": all(
            abs(record["hpac_delta_error_bits"]) < 1e-4 for record in records
        ),
        "mean_hpac_crop_speedup": sum(
            record["hpac_speedup"] for record in records
        ) / len(records),
        "renderer_float_global_in_any_sample": any(
            record["renderer_float_support"]["count"]
            == tokens.shape[1] * tokens.shape[2]
            for record in records
        ),
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
