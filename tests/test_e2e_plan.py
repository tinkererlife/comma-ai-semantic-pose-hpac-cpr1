"""Static integrity checks for the strict raw-video E2E graph."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from e2e import Pipeline  # noqa: E402


def pipeline(tmp_path: Path) -> Pipeline:
    return Pipeline(tmp_path / "challenge", tmp_path / "run", "cuda")


def test_full_graph_is_ordered_non_circular_and_complete(tmp_path):
    value = pipeline(tmp_path)
    assert len(value.stages) == 49
    assert value.stages[0].name == "01_targets"
    assert value.stages[-1].name == "49_official_report"

    produced: set[Path] = set()
    for stage in value.stages:
        for source in stage.inputs:
            if source.is_relative_to(value.run_dir):
                assert source in produced, (
                    f"{stage.name} consumes {source} before it is produced"
                )
        produced.update(stage.outputs)


def test_full_graph_never_consumes_frozen_reference_artifacts(tmp_path):
    value = pipeline(tmp_path)
    frozen = (ROOT / "artifacts").resolve()
    for stage in value.stages:
        for source in stage.inputs:
            assert not source.is_relative_to(frozen), (stage.name, source)
        assert "/artifacts/checkpoints/" not in " ".join(stage.command)
        assert "/artifacts/caches/" not in " ".join(stage.command)


def test_selected_cpu_boundaries_and_fresh_cpr1_mode_are_explicit(tmp_path):
    stages = {stage.name: stage for stage in pipeline(tmp_path).stages}
    cpu_pose = {
        name
        for name, stage in stages.items()
        if name.startswith(tuple(f"{index:02d}_pose" for index in range(9, 33)))
        and "--device" in stage.command
        and stage.command[stage.command.index("--device") + 1] == "cpu"
    }
    assert cpu_pose == {
        "17_pose_cpu_coeff100",
        "18_pose_search32",
        "19_pose_search256",
        "20_pose_cpu_fullqat",
        "24_pose_final_cpu100",
    }
    assert "--allow-noncanonical-source" in stages["46_cpr1"].command
    assert "--require-exact" in stages["44_tokens_verify"].command
    assert stages["48_official_evaluation"].outputs[0].name == "report.txt"
