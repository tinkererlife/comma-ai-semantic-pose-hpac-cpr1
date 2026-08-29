"""Validation tests for completed official challenge reports."""

from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from verify_official_report import parse_report, verify_golden  # noqa: E402


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


def test_golden_report_accepts_expected_band_and_rejects_broken_rail(tmp_path):
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"golden")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    result = parse_report(report(191_052), 191_052)
    verify_golden(
        result,
        archive,
        expected_archive_sha256=digest,
        expected_pose=0.00001981,
        expected_seg=0.00029607,
        expected_score=0.17089548488809853,
        pose_atol=1e-8,
        seg_atol=1e-8,
        score_atol=1e-8,
    )
    broken = dict(result, average_posenet_distortion=0.00054)
    with pytest.raises(ValueError, match="PoseNet distortion"):
        verify_golden(
            broken,
            archive,
            expected_archive_sha256=digest,
            expected_pose=0.00001981,
            expected_seg=0.00029607,
            expected_score=0.17089548488809853,
            pose_atol=1e-5,
            seg_atol=1e-5,
            score_atol=0.01,
        )
