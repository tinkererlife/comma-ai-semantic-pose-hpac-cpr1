#!/usr/bin/env python3
"""Add identity-initialized residual blocks to a semantic renderer checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from semantic_renderer_oracle import SemanticTokenRenderer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--blocks", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    config = dict(checkpoint["config"])
    old_blocks = int(config["blocks"])
    if args.blocks <= old_blocks:
        raise ValueError(
            "expanded block count must exceed the checkpoint block count"
        )
    model = SemanticTokenRenderer(
        width=int(config["width"]),
        blocks=args.blocks,
        frame_dim=int(config["frame_dim"]),
        num_pairs=600,
    )
    expanded = model.state_dict()
    for key, value in checkpoint["state_dict"].items():
        if key not in expanded or expanded[key].shape != value.shape:
            raise ValueError(f"cannot carry checkpoint tensor {key}")
        expanded[key] = value.detach().cpu().clone()
    for block in range(old_blocks, args.blocks):
        expanded[f"blocks.{block}.pw.weight"].zero_()
        expanded[f"blocks.{block}.pw.bias"].zero_()
    config["blocks"] = args.blocks
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": expanded,
        "config": config,
        "expanded_from": str(args.checkpoint),
        "best_exact_seg": checkpoint.get("best_exact_seg"),
    }, args.out)


if __name__ == "__main__":
    main()
