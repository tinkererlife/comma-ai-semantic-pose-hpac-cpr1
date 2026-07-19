#!/usr/bin/env python3
"""Build metric targets through the official CUDA/DALI evaluation path."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--prefetch-queue-depth", type=int, default=4)
    parser.add_argument("--dataset", choices=("dali", "av"), default="dali")
    parser.add_argument("--reference-cache", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("the official target cache requires CUDA")
    device = torch.device("cuda")
    root = args.challenge_root.resolve()
    sys.path.insert(0, str(root))
    from frame_utils import AVVideoDataset, DaliVideoDataset
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path

    names = (root / "public_test_video_names.txt").read_text().splitlines()
    dataset_class = DaliVideoDataset if args.dataset == "dali" else AVVideoDataset
    dataset_device = device if args.dataset == "dali" else torch.device("cpu")
    dataset = dataset_class(
        names,
        data_dir=root / "videos",
        batch_size=args.batch_size,
        device=dataset_device,
        num_threads=args.num_threads,
        seed=1234,
        prefetch_queue_depth=args.prefetch_queue_depth,
    )
    dataset.prepare_data()
    model = DistortionNet().eval().to(device)
    model.load_state_dicts(posenet_sd_path, segnet_sd_path, device)

    poses = []
    segments = []
    started = time.time()
    with torch.inference_mode():
        for _, _, batch in dataset:
            pose, segment = model(batch.to(device).float())
            poses.append(pose["pose"][..., :6].float().cpu())
            segments.append(segment.argmax(1).to(torch.uint8).cpu())
            print(json.dumps({
                "pairs": sum(value.shape[0] for value in poses),
                "elapsed_seconds": time.time() - started,
            }), flush=True)

    result = {"pose": torch.cat(poses), "seg": torch.cat(segments)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.out)

    digest = hashlib.sha256()
    with args.out.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    report = {
        "pairs": result["pose"].shape[0],
        "elapsed_seconds": time.time() - started,
        "cache_bytes": args.out.stat().st_size,
        "cache_sha256": digest.hexdigest(),
        "pose_min": result["pose"].amin(0).tolist(),
        "pose_max": result["pose"].amax(0).tolist(),
    }
    if args.reference_cache is not None:
        reference = torch.load(
            args.reference_cache, map_location="cpu", weights_only=False
        )
        pose_delta = result["pose"] - reference["pose"].float()
        report.update({
            "reference_seg_disagreement": float(
                (result["seg"] != reference["seg"]).float().mean()
            ),
            "reference_pose_mse": float(pose_delta.square().mean()),
            "reference_pose_max_abs": float(pose_delta.abs().max()),
            "reference_pose_dim_mse": pose_delta.square().mean(0).tolist(),
        })
    text = json.dumps(report, indent=2) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text)
    print(text, end="", flush=True)


if __name__ == "__main__":
    main()
