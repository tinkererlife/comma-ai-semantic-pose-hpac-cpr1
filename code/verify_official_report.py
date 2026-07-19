#!/usr/bin/env python3
"""Parse and validate a completed official challenge evaluation report."""

from __future__ import annotations

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    result = parse_report(args.report.read_text(), args.archive.stat().st_size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
