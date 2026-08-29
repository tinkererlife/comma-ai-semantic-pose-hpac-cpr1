#!/usr/bin/env python3
"""Parse and validate a completed official challenge evaluation report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


PATTERNS = {
    "samples": r"Evaluation results over ([0-9,]+) samples",
    "pose": r"Average PoseNet Distortion:\s*([0-9.eE+-]+)",
    "seg": r"Average SegNet Distortion:\s*([0-9.eE+-]+)",
    "archive_bytes": r"Submission file size:\s*([0-9,]+) bytes",
    "original_bytes": r"Original uncompressed size:\s*([0-9,]+) bytes",
    "rate": r"Compression Rate:\s*([0-9.eE+-]+)",
    "displayed_score": r"Final score:.*=\s*([0-9.eE+-]+)",
}


def file_digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def extract(text: str, name: str) -> str:
    match = re.search(PATTERNS[name], text)
    if match is None:
        raise ValueError(f"official report is missing {name}")
    return match.group(1)


def integer(value: str) -> int:
    return int(value.replace(",", ""))


def parse_report(text: str, actual_archive_bytes: int) -> dict[str, object]:
    samples = integer(extract(text, "samples"))
    pose = float(extract(text, "pose"))
    seg = float(extract(text, "seg"))
    archive_bytes = integer(extract(text, "archive_bytes"))
    original_bytes = integer(extract(text, "original_bytes"))
    displayed_rate = float(extract(text, "rate"))
    displayed_score = float(extract(text, "displayed_score"))
    exact_rate = actual_archive_bytes / original_bytes
    exact_score = 100 * seg + math.sqrt(10 * pose) + 25 * exact_rate

    if samples != 600:
        raise ValueError(f"official evaluator covered {samples} samples, not 600")
    if archive_bytes != actual_archive_bytes:
        raise ValueError(
            "official report archive size differs from the staged archive: "
            f"{archive_bytes} != {actual_archive_bytes}"
        )
    if abs(displayed_rate - exact_rate) > 1e-8:
        raise ValueError(
            f"official displayed rate is inconsistent: {displayed_rate} vs {exact_rate}"
        )
    # The evaluator intentionally displays score at low precision.
    if abs(displayed_score - exact_score) > 0.0051:
        raise ValueError(
            "official displayed score is inconsistent with its component metrics: "
            f"{displayed_score} vs {exact_score}"
        )

    return {
        "schema_version": 1,
        "official_samples": samples,
        "average_posenet_distortion": pose,
        "average_segnet_distortion": seg,
        "submission_file_size": actual_archive_bytes,
        "original_uncompressed_size": original_bytes,
        "compression_rate": exact_rate,
        "recomputed_full_precision_score": exact_score,
        "official_displayed_score": displayed_score,
        "report_is_completed": True,
    }


def verify_golden(
    result: dict[str, object],
    archive: Path,
    *,
    expected_archive_sha256: str,
    expected_pose: float,
    expected_seg: float,
    expected_score: float,
    pose_atol: float,
    seg_atol: float,
    score_atol: float,
) -> None:
    actual_sha256 = file_digest(archive)
    if actual_sha256 != expected_archive_sha256:
        raise ValueError(
            "golden archive hash mismatch: "
            f"{actual_sha256} != {expected_archive_sha256}"
        )
    checks = (
        (
            "PoseNet distortion",
            float(result["average_posenet_distortion"]),
            expected_pose,
            pose_atol,
        ),
        (
            "SegNet distortion",
            float(result["average_segnet_distortion"]),
            expected_seg,
            seg_atol,
        ),
        (
            "full-precision score",
            float(result["recomputed_full_precision_score"]),
            expected_score,
            score_atol,
        ),
    )
    failures = [
        f"{name}: {actual} not within {atol} of {expected}"
        for name, actual, expected, atol in checks
        if abs(actual - expected) > atol
    ]
    if failures:
        raise ValueError("golden evaluation mismatch:\n  " + "\n  ".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-archive-sha256")
    parser.add_argument("--expected-pose", type=float)
    parser.add_argument("--expected-seg", type=float)
    parser.add_argument("--expected-score", type=float)
    parser.add_argument("--pose-atol", type=float, default=1e-5)
    parser.add_argument("--seg-atol", type=float, default=1e-5)
    parser.add_argument("--score-atol", type=float, default=0.01)
    args = parser.parse_args()

    result = parse_report(args.report.read_text(), args.archive.stat().st_size)
    expected = (
        args.expected_archive_sha256,
        args.expected_pose,
        args.expected_seg,
        args.expected_score,
    )
    if any(value is not None for value in expected):
        if any(value is None for value in expected):
            raise ValueError("all golden expectations must be supplied together")
        verify_golden(
            result,
            args.archive,
            expected_archive_sha256=args.expected_archive_sha256,
            expected_pose=args.expected_pose,
            expected_seg=args.expected_seg,
            expected_score=args.expected_score,
            pose_atol=args.pose_atol,
            seg_atol=args.seg_atol,
            score_atol=args.score_atol,
        )
        result["golden_archive_passed"] = True
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
