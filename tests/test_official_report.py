"""Validation tests for completed official challenge reports."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from verify_official_report import parse_report  # noqa: E402


def report(archive_bytes: int, samples: int = 600) -> str:
    original = 37_545_489
    rate = archive_bytes / original
    pose = 0.00001981
    seg = 0.00029607
    score = 100 * seg + math.sqrt(10 * pose) + 25 * rate
    return f"""\
=== Evaluation results over {samples} samples ===
  Average PoseNet Distortion: {pose:.8f}
  Average SegNet Distortion: {seg:.8f}
  Submission file size: {archive_bytes:,} bytes
  Original uncompressed size: {original:,} bytes
  Compression Rate: {rate:.8f}
  Final score: 100*segnet_dist + sqrt(10*posenet_dist) + 25*rate = {score:.2f}
"""


def test_completed_official_report_recomputes_full_precision_score():
    result = parse_report(report(191_052), 191_052)
    assert result["official_samples"] == 600
    assert result["submission_file_size"] == 191_052
    assert result["report_is_completed"] is True
    assert result["recomputed_full_precision_score"] == pytest.approx(
        0.17089548488809853
    )


def test_official_report_rejects_partial_or_mismatched_evaluation():
    with pytest.raises(ValueError, match="not 600"):
        parse_report(report(191_052, samples=599), 191_052)
    with pytest.raises(ValueError, match="archive size differs"):
        parse_report(report(191_052), 191_053)
