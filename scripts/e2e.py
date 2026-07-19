#!/usr/bin/env python3
"""Run the selected raw-video-to-CPR1 training and evaluation lineage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
PINNED_CHALLENGE_COMMIT = "d3f688f84f555c5aaebee7d2c4203efc8a9051e2"
RUNTIME_FILES = (
    "archive.zip",
    "inflate.sh",
    "inflate.py",
    "carrier_codec.py",
    "hpac_integer.py",
    "hpac_integer_sparse.py",
    "integer_model_io.py",
)
RUNTIME_PACKAGES = (
    "constriction",
    "nvidia-dali-cuda120",
    "numpy",
    "safetensors",
    "segmentation-models-pytorch",
    "torch",
    "torchvision",
)


@dataclass(frozen=True)
class Stage:
    name: str
    description: str
    command: tuple[str, ...]
    inputs: tuple[Path, ...]
    outputs: tuple[Path, ...]


def path_args(*values: object) -> list[str]:
    return [str(value) for value in values]


def python_command(script: str, *values: object) -> tuple[str, ...]:
    return tuple([sys.executable, str(CODE / script), *path_args(*values)])


def selected_checkpoint(save: Path, kind: str) -> Path:
    if kind == "final":
        return save
    if kind not in {"best", "latest"}:
        raise ValueError(f"unsupported checkpoint selector: {kind}")
    return save.with_name(f"{save.stem}.{kind}.pt")


def file_digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


class Pipeline:
    def __init__(
        self,
        challenge_root: Path,
        run_dir: Path,
        device: str,
    ) -> None:
        self.challenge_root = challenge_root.resolve()
        self.run_dir = run_dir.resolve()
        self.device = device
        self.cache_dir = self.run_dir / "cache"
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.report_dir = self.run_dir / "reports"
        self.artifact_dir = self.run_dir / "artifacts"
        self.log_dir = self.run_dir / "logs"
        self.marker_dir = self.run_dir / ".stages"
        self.target_cache = self.cache_dir / "official_targets.pt"
        self.master_qat12k = self.cache_dir / "masters_qat12k.pt"
        self.master_final = self.cache_dir / "masters_final_semantic.pt"
        self.stages: list[Stage] = []
        self._build()

    def add(
        self,
        name: str,
        description: str,
        command: Iterable[object],
        inputs: Iterable[Path],
        outputs: Iterable[Path],
    ) -> None:
        if any(stage.name == name for stage in self.stages):
            raise ValueError(f"duplicate stage name: {name}")
        self.stages.append(Stage(
            name,
            description,
            tuple(str(value) for value in command),
            tuple(Path(value).resolve() for value in inputs),
            tuple(Path(value).resolve() for value in outputs),
        ))

    def semantic_stage(
        self,
        name: str,
        script: str,
        description: str,
        init: Path | None,
        arguments: list[object],
    ) -> Path:
        save = self.checkpoint_dir / f"{name}.pt"
        report = self.report_dir / f"{name}.json"
        command: list[object] = [
            sys.executable,
            CODE / script,
            "--challenge-root", self.challenge_root,
            "--cache", self.target_cache,
        ]
        inputs = [self.target_cache]
        if init is not None:
            command.extend(["--init", init])
            inputs.append(init)
        command.extend(arguments)
        command.extend(["--device", self.device, "--out", report, "--save", save])
        self.add(name, description, command, inputs, [save, report])
        return save

    def pose_stage(
        self,
        name: str,
        description: str,
        master: Path,
        master_cache: Path,
        init: Path,
        arguments: list[object],
        selected: str = "final",
        creates_master_cache: bool = False,
        device: str | None = None,
    ) -> Path:
        save = self.checkpoint_dir / f"{name}.pt"
        report = self.report_dir / f"{name}.json"
        stage_device = device or self.device
        command: list[object] = [
            sys.executable,
            CODE / "train_pose_carrier_full.py",
            "--challenge-root", self.challenge_root,
            "--target-cache", self.target_cache,
            "--master-checkpoint", master,
            "--init-carrier", init,
            "--master-cache", master_cache,
            "--reuse-master-cache",
            "--cache-masters-on-device",
            *arguments,
            "--device", stage_device,
            "--out", report,
            "--save", save,
        ]
        inputs = [self.target_cache, master, init]
        if not creates_master_cache:
            inputs.append(master_cache)
        outputs = list(dict.fromkeys([
            save, report, selected_checkpoint(save, selected),
        ]))
        if creates_master_cache:
            outputs.append(master_cache)
        self.add(name, description, command, inputs, outputs)
        return selected_checkpoint(save, selected)

    def search_stage(
        self,
        name: str,
        description: str,
        init: Path,
        master_cache: Path,
        arguments: list[object],
        device: str,
    ) -> Path:
        save = self.checkpoint_dir / f"{name}.pt"
        report = self.report_dir / f"{name}.json"
        command: list[object] = [
            sys.executable,
            CODE / "search_pose_coeff_cpu.py",
            "--challenge-root", self.challenge_root,
            "--target-cache", self.target_cache,
            "--master-cache", master_cache,
            "--init", init,
            *arguments,
            "--device", device,
            "--out", report,
            "--save", save,
        ]
        self.add(
            name,
            description,
            command,
            [self.target_cache, master_cache, init],
            [save, report],
        )
        return save

    def refine_stage(
        self,
        name: str,
        description: str,
        init: Path,
        arguments: list[object],
    ) -> Path:
        save = self.checkpoint_dir / f"{name}.pt"
        report = self.report_dir / f"{name}.json"
        command: list[object] = [
            sys.executable,
            CODE / "refine_pose_coeff_codes.py",
            "--challenge-root", self.challenge_root,
            "--target-cache", self.target_cache,
            "--master-cache", self.master_final,
            "--init", init,
            *arguments,
            "--device", self.device,
            "--out", report,
            "--save", save,
        ]
        self.add(
            name,
            description,
            command,
            [self.target_cache, self.master_final, init],
            [save, report],
        )
        return save

    def hpac_stage(
        self,
        name: str,
        description: str,
        arguments: list[object],
        init: Path | None = None,
        selected: str = "final",
    ) -> Path:
        save = self.checkpoint_dir / f"{name}.pt"
        report = self.report_dir / f"{name}.json"
        command: list[object] = [
            sys.executable,
            CODE / "train_hpac_integer.py",
            "--cache", self.target_cache,
        ]
        inputs = [self.target_cache]
        if init is not None:
            command.extend(["--init", init])
            inputs.append(init)
        command.extend(arguments)
        command.extend([
            "--target-mode", "raw",
            "--seed", "20260716",
            "--device", self.device,
            "--save", save,
            "--out", report,
        ])
        outputs = list(dict.fromkeys([
            save, report, selected_checkpoint(save, selected),
        ]))
        self.add(name, description, command, inputs, outputs)
        return selected_checkpoint(save, selected)

    def _build(self) -> None:
        challenge_inputs = [
            self.challenge_root / "public_test_video_names.txt",
            self.challenge_root / "frame_utils.py",
            self.challenge_root / "modules.py",
            self.challenge_root / "models" / "posenet.safetensors",
            self.challenge_root / "models" / "segnet.safetensors",
        ]
        names_path = self.challenge_root / "public_test_video_names.txt"
        if names_path.is_file():
            challenge_inputs.extend(
                self.challenge_root / "videos" / name
                for name in names_path.read_text().splitlines()
                if name
            )
        target_report = self.report_dir / "01_targets.json"
        self.add(
            "01_targets",
            "Extract all 600 official SegNet and PoseNet targets from raw video.",
            python_command(
                "build_gt_cache_official.py",
                "--challenge-root", self.challenge_root,
                "--batch-size", 16,
                "--num-threads", 2,
                "--prefetch-queue-depth", 4,
                "--dataset", "dali",
                "--out", self.target_cache,
                "--report", target_report,
            ),
            challenge_inputs,
            [self.target_cache, target_report],
        )

        semantic_seed = self.semantic_stage(
            "02_semantic_seed_b2",
            "semantic_renderer_oracle.py",
            "Random-init width-96, two-block semantic seed on 12 pairs.",
            None,
            [
                "--pairs", 12,
                "--width", 96,
                "--blocks", 2,
                "--steps", 3000,
                "--lr", 0.001,
                "--frame-dim", 8,
                "--ce-fraction", 0.8,
                "--softplus-fraction", 0.95,
                "--amp",
                "--seed", 20260715,
            ],
        )
        semantic_all = self.semantic_stage(
            "03_semantic_all600",
            "train_semantic_full.py",
            "Expand semantic seed training to all 600 samples.",
            semantic_seed,
            [
                "--steps", 10000,
                "--batch-size", 8,
                "--eval-batch-size", 8,
                "--eval-every", 500,
                "--lr", 0.0005,
                "--ce-fraction", 0.75,
                "--softplus-fraction", 0.92,
                "--amp",
                "--seed", 20260715,
            ],
        )
        semantic_exact = self.semantic_stage(
            "04_semantic_exact_b2",
            "train_semantic_full.py",
            "Optimize the exact camera resize path.",
            semantic_all,
            [
                "--steps", 6000,
                "--batch-size", 2,
                "--eval-batch-size", 8,
                "--eval-every", 250,
                "--lr", 2e-5,
                "--ce-fraction", 0,
                "--softplus-fraction", 0.85,
                "--train-exact-path",
                "--seed", 20260715,
            ],
        )
        semantic_b4_init = self.checkpoint_dir / "05_semantic_expand_b4.pt"
        self.add(
            "05_semantic_expand_b4",
            "Identity-expand the semantic renderer from two to four blocks.",
            python_command(
                "expand_semantic_checkpoint.py",
                "--checkpoint", semantic_exact,
                "--blocks", 4,
                "--out", semantic_b4_init,
            ),
            [semantic_exact],
            [semantic_b4_init],
        )
        semantic_b4 = self.semantic_stage(
            "06_semantic_train_b4",
            "train_semantic_full.py",
            "Train only the two newly inserted semantic residual blocks.",
            semantic_b4_init,
            [
                "--steps", 8000,
                "--batch-size", 2,
                "--eval-batch-size", 8,
                "--eval-every", 250,
                "--lr", 2e-5,
                "--ce-fraction", 0,
                "--softplus-fraction", 0.8,
                "--train-exact-path",
                "--freeze-prefix-blocks", 2,
                "--seed", 20260715,
            ],
        )
        semantic_qat12k = self.semantic_stage(
            "07_semantic_qat12k",
            "train_semantic_quantized.py",
            "Run the selected 4-bit semantic QAT stage.",
            semantic_b4,
            [
                "--bits", 4,
                "--steps", 12000,
                "--batch-size", 2,
                "--eval-batch-size", 8,
                "--eval-every", 250,
                "--lr", 2e-5,
                "--ce-fraction", 0.5,
                "--softplus-fraction", 0.85,
                "--disable-tf32",
                "--seed", 20260715,
            ],
        )
        semantic_final = self.semantic_stage(
            "08_semantic_tail6k",
            "train_semantic_quantized.py",
            "Run the fixed-temperature low-LR 4-bit semantic tail.",
            semantic_qat12k,
            [
                "--bits", 4,
                "--steps", 6000,
                "--batch-size", 2,
                "--eval-batch-size", 8,
                "--eval-every", 250,
                "--lr", 2e-7,
                "--ce-fraction", 0,
                "--softplus-fraction", -999,
                "--disable-tf32",
                "--seed", 20260716,
            ],
        )

        pilot4 = self.checkpoint_dir / "09_pose_pilot4.pt"
        pilot4_report = self.report_dir / "09_pose_pilot4.json"
        self.add(
            "09_pose_pilot4",
            "Learn the first four-pair, eight-direction pose basis.",
            python_command(
                "learned_pose_carrier_oracle.py",
                "--challenge-root", self.challenge_root,
                "--target-cache", self.target_cache,
                "--master-checkpoint", semantic_seed,
                "--pairs", 4,
                "--basis-dim", 8,
                "--basis-height", 24,
                "--basis-width", 32,
                "--steps", 1600,
                "--lr", 0.005,
                "--amplitude", 64,
                "--carrier-base", "gray",
                "--basis-bits", 8,
                "--qat-fraction", 0.6,
                "--seed", 20260715,
                "--device", self.device,
                "--out", pilot4_report,
                "--save", pilot4,
            ),
            [self.target_cache, semantic_seed],
            [pilot4, pilot4_report],
        )
        pilot12 = self.checkpoint_dir / "10_pose_pilot12.pt"
        pilot12_report = self.report_dir / "10_pose_pilot12.json"
        self.add(
            "10_pose_pilot12",
            "Expand the pose pilot to 12 directions and 12 sampled pairs.",
            python_command(
                "learned_pose_carrier_oracle.py",
                "--challenge-root", self.challenge_root,
                "--target-cache", self.target_cache,
                "--master-checkpoint", semantic_seed,
                "--pairs", 12,
                "--basis-dim", 12,
                "--basis-height", 24,
                "--basis-width", 32,
                "--steps", 2200,
                "--lr", 0.005,
                "--amplitude", 64,
                "--carrier-base", "gray",
                "--basis-bits", 8,
                "--qat-fraction", 0.7,
                "--resume-basis", pilot4,
                "--seed", 20260715,
                "--device", self.device,
                "--out", pilot12_report,
                "--save", pilot12,
            ),
            [self.target_cache, semantic_seed, pilot4],
            [pilot12, pilot12_report],
        )

        pose_joint = self.pose_stage(
            "11_pose_joint_qat",
            "Initialize all 600 coefficient rows with row-local QAT.",
            semantic_qat12k,
            self.master_qat12k,
            pilot12,
            [
                "--steps", 10000,
                "--batch-size", 128,
                "--eval-batch-size", 16,
                "--render-batch-size", 4,
                "--eval-every", 500,
                "--lr-basis", 0.001,
                "--lr-coeff", 0.005,
                "--basis-freeze-fraction", 0,
                "--basis-train-until-fraction", 0.25,
                "--qat-fraction", 0,
                "--coeff-qat-fraction", 0,
                "--metric-loss-after-basis",
                "--metric-normalized-weight", 0,
                "--basis-bits", 8,
                "--coeff-bits", 12,
                "--amplitude", 64,
                "--master-carrier-amplitude", 0,
                "--carrier-base", "gray",
                "--seed", 20260715,
            ],
            selected="latest",
            creates_master_cache=True,
        )
        pose_raw = self.pose_stage(
            "12_pose_raw7500",
            "Run the selected frozen-basis raw-metric continuation.",
            semantic_qat12k,
            self.master_qat12k,
            pose_joint,
            [
                "--steps", 7500,
                "--batch-size", 128,
                "--eval-batch-size", 16,
                "--render-batch-size", 4,
                "--eval-every", 500,
                "--lr-basis", 1e-6,
                "--lr-coeff", 0.001,
                "--basis-freeze-fraction", 1,
                "--basis-train-until-fraction", 0,
                "--qat-fraction", 0,
                "--coeff-qat-fraction", 0,
                "--always-metric-loss",
                "--basis-bits", 8,
                "--coeff-bits", 12,
                "--amplitude", 64,
                "--master-carrier-amplitude", 0,
                "--carrier-base", "gray",
                "--seed", 20260715,
            ],
            selected="latest",
        )
        pose_hard = self.pose_stage(
            "13_pose_hard750",
            "Reproduce the selected step-750 hard-mining checkpoint.",
            semantic_qat12k,
            self.master_qat12k,
            pose_raw,
            [
                "--steps", 4000,
                "--stop-after-step", 750,
                "--batch-size", 128,
                "--eval-batch-size", 16,
                "--render-batch-size", 4,
                "--eval-every", 250,
                "--lr-basis", 1e-6,
                "--lr-coeff", 2e-4,
                "--basis-freeze-fraction", 1,
                "--basis-train-until-fraction", 0,
                "--qat-fraction", 0,
                "--coeff-qat-fraction", 0,
                "--always-metric-loss",
                "--hard-mining-power", 0.5,
                "--hard-mining-max", 8,
                "--basis-bits", 8,
                "--coeff-bits", 12,
                "--amplitude", 64,
                "--master-carrier-amplitude", 0,
                "--carrier-base", "gray",
                "--seed", 20260715,
            ],
            selected="latest",
        )
        pose_resident = self.pose_stage(
            "14_pose_resident",
            "Continue the selected hard-mining carrier stage.",
            semantic_qat12k,
            self.master_qat12k,
            pose_hard,
            [
                "--steps", 4000,
                "--batch-size", 128,
                "--eval-batch-size", 16,
                "--render-batch-size", 4,
                "--eval-every", 250,
                "--lr-basis", 1e-6,
                "--lr-coeff", 2e-4,
                "--basis-freeze-fraction", 1,
                "--basis-train-until-fraction", 0,
                "--qat-fraction", 0,
                "--coeff-qat-fraction", 0,
                "--always-metric-loss",
                "--hard-mining-power", 0.5,
                "--hard-mining-max", 8,
                "--basis-bits", 8,
                "--coeff-bits", 12,
                "--amplitude", 64,
                "--master-carrier-amplitude", 0,
                "--carrier-base", "gray",
                "--seed", 20260715,
            ],
            selected="best",
        )
        pose_uniform = self.pose_stage(
            "15_pose_uniform",
            "Remove hard-mining bias with the selected uniform continuation.",
            semantic_qat12k,
            self.master_qat12k,
            pose_resident,
            [
                "--steps", 4000,
                "--batch-size", 128,
                "--eval-batch-size", 16,
                "--render-batch-size", 4,
                "--eval-every", 250,
                "--lr-basis", 1e-6,
                "--lr-coeff", 1e-4,
                "--basis-freeze-fraction", 1,
                "--basis-train-until-fraction", 0,
                "--qat-fraction", 0,
                "--coeff-qat-fraction", 0,
                "--always-metric-loss",
                "--hard-mining-power", 0,
                "--basis-bits", 8,
                "--coeff-bits", 12,
                "--amplitude", 64,
                "--master-carrier-amplitude", 0,
                "--carrier-base", "gray",
                "--seed", 20260715,
            ],
            selected="best",
        )
        pose_tail64 = self.pose_stage(
            "16_pose_tail64",
            "Polish the 64 hardest coefficient rows.",
            semantic_qat12k,
            self.master_qat12k,
            pose_uniform,
            [
                "--steps", 2000,
                "--batch-size", 128,
                "--eval-batch-size", 16,
                "--render-batch-size", 4,
                "--eval-every", 250,
                "--lr-basis", 1e-6,
                "--lr-coeff", 1e-4,
                "--basis-freeze-fraction", 1,
                "--basis-train-until-fraction", 0,
                "--qat-fraction", 0,
                "--coeff-qat-fraction", 0,
                "--always-metric-loss",
                "--hard-mining-power", 1,
                "--hard-mining-max", 64,
                "--basis-bits", 8,
                "--coeff-bits", 12,
                "--amplitude", 64,
                "--master-carrier-amplitude", 0,
                "--carrier-base", "gray",
                "--seed", 20260715,
            ],
            selected="final",
        )
        pose_cpu = self.pose_stage(
            "17_pose_cpu_coeff100",
            "Run the selected deterministic CPU coefficient polish.",
            semantic_qat12k,
            self.master_qat12k,
            pose_tail64,
            [
                "--steps", 100,
                "--batch-size", 16,
                "--eval-batch-size", 16,
                "--render-batch-size", 4,
                "--eval-every", 20,
                "--lr-basis", 0,
                "--lr-coeff", 1e-4,
                "--basis-freeze-fraction", 1,
                "--basis-train-until-fraction", 0,
                "--qat-fraction", 0,
                "--coeff-qat-fraction", 0,
                "--always-metric-loss",
                "--basis-bits", 8,
                "--coeff-bits", 12,
                "--amplitude", 64,
                "--master-carrier-amplitude", 0,
                "--carrier-base", "gray",
                "--seed", 20260715,
            ],
            selected="final",
            device="cpu",
        )
        pose_search1 = self.search_stage(
            "18_pose_search32",
            "Search exact int12 neighbors through code step 32.",
            pose_cpu,
            self.master_qat12k,
            [
                "--top-k", 1,
                "--passes", 1,
                "--steps", 1, 2, 4, 8, 16, 32,
                "--eval-batch-size", 8,
                "--candidate-batch-size", 8,
                "--amplitude", 64,
            ],
            "cpu",
        )
        pose_search2 = self.search_stage(
            "19_pose_search256",
            "Widen exact int12 neighbor search through code step 256.",
            pose_search1,
            self.master_qat12k,
            [
                "--top-k", 1,
                "--passes", 1,
                "--steps", 1, 2, 4, 8, 16, 32, 64, 128, 256,
                "--eval-batch-size", 8,
                "--candidate-batch-size", 8,
                "--amplitude", 64,
            ],
            "cpu",
        )
        pose_cpu_full = self.pose_stage(
            "20_pose_cpu_fullqat",
            "Run joint CPU basis and coefficient QAT before retargeting.",
            semantic_qat12k,
            self.master_qat12k,
            pose_search2,
            [
                "--steps", 400,
                "--batch-size", 16,
                "--eval-batch-size", 16,
                "--render-batch-size", 4,
                "--eval-every", 50,
                "--lr-basis", 1e-6,
                "--lr-coeff", 1e-4,
                "--basis-freeze-fraction", 0,
                "--basis-train-until-fraction", 1,
                "--qat-fraction", 0.5,
                "--coeff-qat-fraction", 0.5,
                "--always-metric-loss",
                "--basis-bits", 8,
                "--coeff-bits", 12,
                "--amplitude", 64,
                "--master-carrier-amplitude", 0,
                "--carrier-base", "gray",
                "--seed", 20260715,
            ],
            selected="final",
            device="cpu",
        )
        pose_retarget = self.pose_stage(
            "21_pose_retarget_coeff1000",
            "Retarget coefficients to the final semantic renderer at step 1000.",
            semantic_final,
            self.master_final,
            pose_cpu_full,
            [
                "--steps", 2000,
                "--stop-after-step", 1000,
                "--batch-size", 128,
                "--eval-batch-size", 16,
                "--render-batch-size", 4,
                "--eval-every", 250,
                "--lr-basis", 1e-6,
                "--lr-coeff", 0.01,
                "--basis-freeze-fraction", 1,
                "--basis-train-until-fraction", 0,
                "--qat-fraction", 0.5,
                "--coeff-qat-fraction", 0.5,
                "--always-metric-loss",
                "--hard-mining-power", 1,
                "--hard-mining-max", 64,
                "--basis-bits", 8,
                "--coeff-bits", 12,
                "--amplitude", 64,
                "--master-carrier-amplitude", 0,
                "--carrier-base", "gray",
                "--seed", 20260718,
            ],
            selected="latest",
            creates_master_cache=True,
        )
        pose_adapt250 = self.pose_stage(
            "22_pose_basis_adapt250",
            "Reproduce the selected step-250 basis adaptation.",
            semantic_final,
            self.master_final,
            pose_retarget,
            [
                "--steps", 2000,
                "--stop-after-step", 250,
                "--batch-size", 128,
                "--eval-batch-size", 16,
                "--render-batch-size", 4,
                "--eval-every", 250,
                "--lr-basis", 1e-4,
                "--lr-coeff", 0.001,
                "--basis-freeze-fraction", 0,
                "--basis-train-until-fraction", 1,
                "--qat-fraction", 0,
                "--coeff-qat-fraction", 0,
                "--always-metric-loss",
                "--hard-mining-power", 0.5,
                "--hard-mining-max", 16,
                "--basis-bits", 8,
                "--coeff-bits", 12,
                "--amplitude", 64,
                "--master-carrier-amplitude", 0,
                "--carrier-base", "gray",
                "--seed", 20260719,
            ],
            selected="best",
        )
        pose_adapt = self.pose_stage(
            "23_pose_basis_adapt3000",
            "Run the selected low-LR full-basis adaptation.",
            semantic_final,
            self.master_final,
            pose_adapt250,
            [
                "--steps", 3000,
                "--batch-size", 128,
                "--eval-batch-size", 16,
                "--render-batch-size", 4,
                "--eval-every", 250,
                "--lr-basis", 2e-5,
                "--lr-coeff", 2e-4,
                "--basis-freeze-fraction", 0,
                "--basis-train-until-fraction", 1,
                "--qat-fraction", 0,
                "--coeff-qat-fraction", 0,
                "--always-metric-loss",
                "--hard-mining-power", 0,
                "--basis-bits", 8,
                "--coeff-bits", 12,
                "--amplitude", 64,
                "--master-carrier-amplitude", 0,
                "--carrier-base", "gray",
                "--seed", 20260720,
            ],
            selected="final",
        )
        pose_final_cpu = self.pose_stage(
            "24_pose_final_cpu100",
            "Reproduce the selected final-semantic CPU step-100 checkpoint.",
            semantic_final,
            self.master_final,
            pose_adapt,
            [
                "--steps", 400,
                "--stop-after-step", 100,
                "--batch-size", 16,
                "--eval-batch-size", 16,
                "--render-batch-size", 4,
                "--eval-every", 50,
                "--lr-basis", 2e-7,
                "--lr-coeff", 2e-5,
                "--basis-freeze-fraction", 0,
                "--basis-train-until-fraction", 1,
                "--qat-fraction", 0,
                "--coeff-qat-fraction", 0,
                "--always-metric-loss",
                "--basis-bits", 8,
                "--coeff-bits", 12,
                "--amplitude", 64,
                "--master-carrier-amplitude", 0,
                "--carrier-base", "gray",
                "--seed", 20260722,
            ],
            selected="best",
            device="cpu",
        )
        pose_official = self.pose_stage(
            "25_pose_official_coeff",
            "Run the official-cache coefficient rail and select its best checkpoint.",
            semantic_final,
            self.master_final,
            pose_final_cpu,
            [
                "--steps", 2000,
                "--batch-size", 64,
                "--eval-batch-size", 16,
                "--render-batch-size", 4,
                "--eval-every", 100,
                "--lr-basis", 1e-8,
                "--lr-coeff", 0.01,
                "--basis-freeze-fraction", 1,
                "--basis-train-until-fraction", 0,
                "--qat-fraction", 0,
                "--coeff-qat-fraction", 0,
                "--always-metric-loss",
                "--basis-bits", 8,
                "--coeff-bits", 12,
                "--amplitude", 64,
                "--master-carrier-amplitude", 0,
                "--carrier-base", "gray",
                "--seed", 20260716,
            ],
            selected="best",
        )
        pose_grid1 = self.search_stage(
            "26_pose_grid128x3",
            "Run three exact int12 coordinate-search passes over 128 rows.",
            pose_official,
            self.master_final,
            [
                "--top-k", 128,
                "--passes", 3,
                "--steps", 1, 2, 4, 8, 16, 32, 64, 128,
                "--eval-batch-size", 8,
                "--candidate-batch-size", 8,
                "--amplitude", 64,
            ],
            self.device,
        )
        pose_grid2 = self.search_stage(
            "27_pose_grid64x12",
            "Run 12 wide exact int12 coordinate-search passes over 64 rows.",
            pose_grid1,
            self.master_final,
            [
                "--top-k", 64,
                "--passes", 12,
                "--steps", 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                "--eval-batch-size", 8,
                "--candidate-batch-size", 8,
                "--amplitude", 64,
            ],
            self.device,
        )
        pose_refine1 = self.refine_stage(
            "28_pose_refine_pass1",
            "Refine 128 exact int12 rows with protected scale anchors.",
            pose_grid2,
            [
                "--top-k", 128,
                "--steps", 1000,
                "--train-batch-size", 8,
                "--eval-batch-size", 8,
                "--eval-every", 20,
                "--lr", 8,
                "--amplitude", 64,
                "--basis-bits", 8,
                "--disable-tf32",
                "--seed", 20260716,
            ],
        )
        pose_refine2 = self.refine_stage(
            "29_pose_refine_pass2",
            "Run the second exact-code refinement pass.",
            pose_refine1,
            [
                "--top-k", 128,
                "--steps", 1200,
                "--train-batch-size", 8,
                "--eval-batch-size", 8,
                "--eval-every", 20,
                "--lr", 4,
                "--amplitude", 64,
                "--basis-bits", 8,
                "--disable-tf32",
                "--seed", 20260717,
            ],
        )
        pose_anchor = self.refine_stage(
            "30_pose_anchor",
            "Run the selected anchor-preserving exact-code pass.",
            pose_refine2,
            [
                "--top-k", 128,
                "--steps", 1000,
                "--train-batch-size", 8,
                "--eval-batch-size", 8,
                "--eval-every", 20,
                "--lr", 4,
                "--amplitude", 64,
                "--basis-bits", 8,
                "--disable-tf32",
                "--seed", 20260719,
            ],
        )
        pose_int6 = self.pose_stage(
            "31_pose_int6_stable8k",
            "Migrate and stabilize the carrier basis at six bits.",
            semantic_final,
            self.master_final,
            pose_anchor,
            [
                "--steps", 8000,
                "--batch-size", 4,
                "--eval-batch-size", 8,
                "--render-batch-size", 4,
                "--eval-every", 500,
                "--lr-basis", 1e-4,
                "--lr-coeff", 0.003,
                "--basis-freeze-fraction", 0.25,
                "--basis-train-until-fraction", 0.75,
                "--qat-fraction", 0,
                "--coeff-qat-fraction", 0,
                "--always-metric-loss",
                "--hard-mining-power", 0.5,
                "--hard-mining-max", 4,
                "--basis-bits", 6,
                "--coeff-bits", 12,
                "--amplitude", 64,
                "--master-carrier-amplitude", 0,
                "--carrier-base", "gray",
                "--seed", 20260718,
            ],
            selected="final",
        )
        pose_final = self.pose_stage(
            "32_pose_int6_coefftail4k",
            "Run the selected frozen-basis int6 coefficient tail.",
            semantic_final,
            self.master_final,
            pose_int6,
            [
                "--steps", 4000,
                "--batch-size", 8,
                "--eval-batch-size", 8,
                "--render-batch-size", 4,
                "--eval-every", 250,
                "--lr-basis", 1e-6,
                "--lr-coeff", 3e-4,
                "--basis-freeze-fraction", 1,
                "--basis-train-until-fraction", 0,
                "--qat-fraction", 0,
                "--coeff-qat-fraction", 0,
                "--always-metric-loss",
                "--hard-mining-power", 0.75,
                "--hard-mining-max", 6,
                "--basis-bits", 6,
                "--coeff-bits", 12,
                "--amplitude", 64,
                "--master-carrier-amplitude", 0,
                "--carrier-base", "gray",
                "--seed", 20260722,
            ],
            selected="final",
        )

        hpac_smoke = self.hpac_stage(
            "33_hpac_smoke",
            "Random-init exact-integer HPAC smoke stage.",
            [
                "--epochs", 5,
                "--batch-size", 8,
                "--eval-batch-size", 4,
                "--eval-every", 1,
                "--lr", 0.02,
                "--channels", 64,
                "--patch", 32,
                "--delta", 2,
                "--frame-dim", 8,
            ],
        )
        hpac_epoch60 = self.hpac_stage(
            "34_hpac_epoch60",
            "Reproduce the selected epoch-60 point on a 100-epoch schedule.",
            [
                "--epochs", 100,
                "--stop-after-epoch", 60,
                "--batch-size", 8,
                "--eval-batch-size", 4,
                "--eval-every", 10,
                "--lr", 0.01,
                "--channels", 64,
                "--patch", 32,
                "--delta", 2,
                "--frame-dim", 8,
            ],
            hpac_smoke,
            selected="latest",
        )
        hpac_raw = self.hpac_stage(
            "35_hpac_raw_refine",
            "Run the selected raw-token integer HPAC refinement.",
            [
                "--epochs", 200,
                "--batch-size", 8,
                "--eval-batch-size", 4,
                "--eval-every", 10,
                "--lr", 0.005,
                "--channels", 64,
                "--patch", 32,
                "--delta", 2,
                "--frame-dim", 8,
            ],
            hpac_epoch60,
        )
        hpac_spm = self.hpac_stage(
            "36_hpac_spm",
            "Add and train the spatial-pyramid module.",
            [
                "--epochs", 80,
                "--batch-size", 8,
                "--eval-batch-size", 4,
                "--eval-every", 2,
                "--lr", 0.003,
                "--lr-spm", 0.05,
                "--channels", 64,
                "--patch", 32,
                "--delta", 2,
                "--frame-dim", 8,
                "--spm",
            ],
            hpac_raw,
        )
        hpac_frame = self.hpac_stage(
            "37_hpac_frame_scale",
            "Add and train only the per-frame scale carrier.",
            [
                "--epochs", 60,
                "--batch-size", 8,
                "--eval-batch-size", 4,
                "--eval-every", 2,
                "--lr", 0,
                "--lr-spm", 0,
                "--lr-frame-scale", 0.05,
                "--channels", 64,
                "--patch", 32,
                "--delta", 2,
                "--frame-dim", 8,
                "--spm",
                "--frame-scale",
            ],
            hpac_spm,
        )
        hpac_migrate = self.hpac_stage(
            "38_hpac_halfstep_migrate",
            "Migrate integer weights onto the selected dyadic half-step lattice.",
            [
                "--epochs", 1,
                "--batch-size", 8,
                "--eval-batch-size", 4,
                "--eval-every", 1,
                "--lr", 0,
                "--lr-weight", 0.04,
                "--lr-exponent", 0,
                "--lr-spm", 0,
                "--lr-frame-scale", 0,
                "--channels", 64,
                "--patch", 32,
                "--delta", 2,
                "--frame-dim", 8,
                "--spm",
                "--frame-scale",
                "--weight-scales",
                "--migration-exponent", -1,
                "--weight-exponent-min", -6,
                "--weight-bound", 127,
                "--activation-bound", 127,
            ],
            hpac_frame,
        )
        hpac_half = self.hpac_stage(
            "39_hpac_halfstep_refine",
            "Refine weights while retaining the selected dyadic lattice.",
            [
                "--epochs", 50,
                "--batch-size", 8,
                "--eval-batch-size", 4,
                "--eval-every", 2,
                "--lr", 0,
                "--lr-weight", 0.02,
                "--lr-exponent", 0,
                "--lr-spm", 0,
                "--lr-frame-scale", 0,
                "--channels", 64,
                "--patch", 32,
                "--delta", 2,
                "--frame-dim", 8,
                "--spm",
                "--frame-scale",
                "--weight-scales",
                "--weight-exponent-min", -6,
                "--weight-bound", 127,
                "--activation-bound", 127,
            ],
            hpac_migrate,
        )
        hpac_p64 = self.hpac_stage(
            "40_hpac_patch64",
            "Migrate and tune the final 64x64 integer HPAC.",
            [
                "--epochs", 40,
                "--batch-size", 8,
                "--eval-batch-size", 4,
                "--eval-every", 2,
                "--lr", 0.002,
                "--lr-weight", 0.01,
                "--lr-exponent", 0,
                "--lr-frame-scale", 0.005,
                "--lr-spm", 0.005,
                "--channels", 64,
                "--patch", 64,
                "--delta", 2,
                "--frame-dim", 8,
                "--spm",
                "--frame-scale",
                "--weight-scales",
                "--weight-exponent-min", -6,
                "--weight-bound", 127,
                "--activation-bound", 127,
            ],
            hpac_half,
        )
        hpac_self = self.checkpoint_dir / "41_hpac_selfcompress.pt"
        hpac_self_report = self.report_dir / "41_hpac_selfcompress.json"
        self.add(
            "41_hpac_selfcompress",
            "Jointly optimize arithmetic-token rate and self-compressed model rate.",
            python_command(
                "train_hpac_self_compress.py",
                "--cache", self.target_cache,
                "--init", hpac_p64,
                "--epochs", 60,
                "--batch-size", 8,
                "--eval-batch-size", 4,
                "--eval-every", 2,
                "--lr", 0.003,
                "--lr-exponent", 0.0002,
                "--lr-bits", 0.01,
                "--bit-eps", 1e-6,
                "--rate-lambda", 1,
                "--qat-fraction", 0.5,
                "--init-bits", 8,
                "--channels", 64,
                "--patch", 64,
                "--delta", 2,
                "--frame-dim", 8,
                "--norm-mode", "none",
                "--activation", "relu",
                "--frame-scale",
                "--weight-bound", 127,
                "--activation-bound", 127,
                "--weight-scales",
                "--weight-exponent-min", -6,
                "--spm",
                "--target-mode", "raw",
                "--seed", 20260716,
                "--device", self.device,
                "--save", hpac_self,
                "--out", hpac_self_report,
            ),
            [self.target_cache, hpac_p64],
            [hpac_self, hpac_self_report],
        )
        hpac_blob = self.artifact_dir / "hpac.bin.xz"
        hpac_pack_report = self.report_dir / "42_hpac_pack.json"
        self.add(
            "42_hpac_pack",
            "Pack and exact-round-trip the self-compressed integer HPAC.",
            python_command(
                "pack_hpac_self_compress.py",
                "--checkpoint", hpac_self,
                "--channels", 64,
                "--patch", 64,
                "--delta", 2,
                "--frame-dim", 8,
                "--weight-bound", 127,
                "--activation-bound", 127,
                "--weight-exponent-min", -6,
                "--device", "cpu",
                "--blob", hpac_blob,
                "--report", hpac_pack_report,
            ),
            [hpac_self],
            [hpac_blob, hpac_pack_report],
        )
        tokens = self.artifact_dir / "tokens.bin"
        encode_report = self.report_dir / "43_tokens_encode.json"
        codec_common: list[object] = [
            "--checkpoint", hpac_self,
            "--cache", self.target_cache,
            "--channels", 64,
            "--patch", 64,
            "--delta", 2,
            "--frame-dim", 8,
            "--norm-mode", "none",
            "--activation", "relu",
            "--frame-scale",
            "--weight-bound", 127,
            "--activation-bound", 127,
            "--weight-scales",
            "--weight-exponent-min", -6,
            "--spm",
            "--sparse",
            "--self-compress",
            "--target-mode", "raw",
            "--frames", 600,
            "--device", self.device,
        ]
        self.add(
            "43_tokens_encode",
            "Arithmetic-encode all 600 exact semantic maps.",
            python_command(
                "codec_hpac_integer.py",
                *codec_common,
                "--tokens-out", tokens,
                "--report", encode_report,
            ),
            [hpac_self, self.target_cache],
            [tokens, encode_report],
        )
        decode_report = self.report_dir / "44_tokens_decode.json"
        self.add(
            "44_tokens_verify",
            "Decode the token stream and require exact equality to fresh targets.",
            python_command(
                "codec_hpac_integer.py",
                *codec_common,
                "--decode-from", tokens,
                "--require-exact",
                "--report", decode_report,
            ),
            [hpac_self, self.target_cache, tokens],
            [decode_report],
        )

        predecessor = self.artifact_dir / "predecessor"
        predecessor_report = self.report_dir / "45_predecessor.json"
        self.add(
            "45_predecessor",
            "Pack fresh semantic, carrier, HPAC, and tokens into the legacy archive.",
            python_command(
                "build_submission_archive.py",
                "--semantic", semantic_final,
                "--carrier", pose_final,
                "--hpac", hpac_blob,
                "--tokens", tokens,
                "--basis-bits", 5,
                "--submission-dir", predecessor,
                "--report", predecessor_report,
            ),
            [semantic_final, pose_final, hpac_blob, tokens],
            [
                predecessor / "semantic_pose.bin.xz",
                predecessor / "p",
                predecessor / "archive.zip",
                predecessor_report,
            ],
        )
        cpr1_archive = self.artifact_dir / "cpr1" / "archive.zip"
        cpr1_report = self.report_dir / "46_cpr1.json"
        self.add(
            "46_cpr1",
            "Losslessly repack the fresh legacy carrier as canonical CPR1.",
            python_command(
                "repack_carrier.py",
                predecessor / "archive.zip",
                cpr1_archive,
                "--allow-noncanonical-source",
                "--report", cpr1_report,
            ),
            [predecessor / "archive.zip"],
            [cpr1_archive, cpr1_report],
        )
        submission = self.run_dir / "submission"
        stage_report = self.report_dir / "47_submission.json"
        self.add(
            "47_submission",
            "Stage only the charged CPR1 archive and its minimal inflate runtime.",
            python_command(
                "stage_submission.py",
                "--archive", cpr1_archive,
                "--code-root", CODE,
                "--submission-dir", submission,
                "--report", stage_report,
            ),
            [
                cpr1_archive,
                *(CODE / name for name in RUNTIME_FILES if name != "archive.zip"),
            ],
            [
                *(submission / name for name in RUNTIME_FILES),
                stage_report,
            ],
        )
        official_report = submission / "report.txt"
        self.add(
            "48_official_evaluation",
            "Inflate CPR1 and score all 600 samples with the official evaluator.",
            (
                "bash",
                str(self.challenge_root / "evaluate.sh"),
                "--submission-dir", str(submission),
                "--device", self.device,
            ),
            [
                *(submission / name for name in RUNTIME_FILES),
                *challenge_inputs,
                self.challenge_root / "evaluate.sh",
                self.challenge_root / "evaluate.py",
            ],
            [official_report],
        )
        official_metrics = self.report_dir / "49_official_metrics.json"
        self.add(
            "49_official_report",
            "Parse the completed official report and recompute full-precision score.",
            python_command(
                "verify_official_report.py",
                "--report", official_report,
                "--archive", submission / "archive.zip",
                "--out", official_metrics,
            ),
            [official_report, submission / "archive.zip"],
            [official_metrics],
        )


class Runner:
    def __init__(self, pipeline: Pipeline, force: bool) -> None:
        self.pipeline = pipeline
        self.force = force
        self._fingerprint_cache: dict[
            tuple[str, int, int], dict[str, object]
        ] = {}
        self.pipeline_hash = self._pipeline_hash()
        self.runtime_identity = runtime_identity()

    def _pipeline_hash(self) -> str:
        value = hashlib.sha256()
        files = sorted(CODE.glob("*.py")) + [
            CODE / "inflate.sh",
            Path(__file__).resolve(),
        ]
        for path in files:
            value.update(str(path.relative_to(ROOT)).encode())
            value.update(path.read_bytes())
        return value.hexdigest()

    def fingerprint(self, path: Path) -> dict[str, object]:
        path = path.resolve()
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        if key not in self._fingerprint_cache:
            self._fingerprint_cache[key] = {
                "path": str(path),
                "bytes": stat.st_size,
                "sha256": file_digest(path),
            }
        return self._fingerprint_cache[key]

    def marker_path(self, stage: Stage) -> Path:
        return self.pipeline.marker_dir / f"{stage.name}.json"

    def marker_is_valid(self, stage: Stage) -> bool:
        marker_path = self.marker_path(stage)
        if self.force or not marker_path.is_file():
            return False
        try:
            marker = json.loads(marker_path.read_text())
            if marker["command"] != list(stage.command):
                return False
            if marker["pipeline_sha256"] != self.pipeline_hash:
                return False
            if marker["runtime_identity"] != self.runtime_identity:
                return False
            inputs = [self.fingerprint(path) for path in stage.inputs]
            outputs = [self.fingerprint(path) for path in stage.outputs]
            return marker["inputs"] == inputs and marker["outputs"] == outputs
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            return False

    def run_stage(self, stage: Stage) -> None:
        if self.marker_is_valid(stage):
            print(f"SKIP {stage.name}: verified marker and outputs", flush=True)
            return
        missing = [path for path in stage.inputs if not path.is_file()]
        if missing:
            paths = "\n".join(f"  {path}" for path in missing)
            raise FileNotFoundError(
                f"{stage.name} is missing required inputs:\n{paths}"
            )
        for output in stage.outputs:
            output.parent.mkdir(parents=True, exist_ok=True)
        self.pipeline.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.pipeline.log_dir / f"{stage.name}.log"
        print(f"\nRUN  {stage.name}: {stage.description}", flush=True)
        print(f"     {shlex.join(stage.command)}", flush=True)
        environment = os.environ.copy()
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            str(CODE)
            if not existing_pythonpath
            else f"{CODE}{os.pathsep}{existing_pythonpath}"
        )
        environment.setdefault("PYTHONHASHSEED", "0")
        environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        started = time.time()
        with log_path.open("w") as log:
            process = subprocess.Popen(
                stage.command,
                cwd=ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            return_code = process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, stage.command)
        missing = [path for path in stage.outputs if not path.is_file()]
        if missing:
            paths = "\n".join(f"  {path}" for path in missing)
            raise RuntimeError(
                f"{stage.name} completed without required outputs:\n{paths}"
            )
        marker = {
            "schema_version": 1,
            "stage": stage.name,
            "description": stage.description,
            "command": list(stage.command),
            "pipeline_sha256": self.pipeline_hash,
            "runtime_identity": self.runtime_identity,
            "elapsed_seconds": time.time() - started,
            "inputs": [self.fingerprint(path) for path in stage.inputs],
            "outputs": [self.fingerprint(path) for path in stage.outputs],
            "log": str(log_path),
        }
        self.pipeline.marker_dir.mkdir(parents=True, exist_ok=True)
        self.marker_path(stage).write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n"
        )
        print(
            f"DONE {stage.name} ({marker['elapsed_seconds']:.1f}s)",
            flush=True,
        )

    def write_manifest(self, stages: list[Stage]) -> None:
        stage_records = []
        for stage in stages:
            marker = self.marker_path(stage)
            if marker.is_file():
                stage_records.append(json.loads(marker.read_text()))
        archive = self.pipeline.run_dir / "submission" / "archive.zip"
        report = self.pipeline.run_dir / "submission" / "report.txt"
        official_metrics = (
            self.pipeline.report_dir / "49_official_metrics.json"
        )
        manifest = {
            "schema_version": 1,
            "pipeline_sha256": self.pipeline_hash,
            "runtime_identity": self.runtime_identity,
            "challenge_commit": git_commit(self.pipeline.challenge_root),
            "stages": stage_records,
            "final_archive": (
                self.fingerprint(archive) if archive.is_file() else None
            ),
            "official_report": str(report) if report.is_file() else None,
            "official_metrics": (
                json.loads(official_metrics.read_text())
                if official_metrics.is_file() else None
            ),
        }
        (self.pipeline.run_dir / "e2e-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )


def git_commit(repository: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def runtime_identity() -> dict[str, object]:
    packages = {}
    for package in RUNTIME_PACKAGES:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    gpu = None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        gpu = result.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "gpu": gpu,
    }


def frozen_guard(stages: list[Stage]) -> None:
    frozen_root = (ROOT / "artifacts").resolve()
    violations = []
    for stage in stages:
        for path in stage.inputs:
            try:
                path.resolve().relative_to(frozen_root)
            except ValueError:
                continue
            violations.append((stage.name, path))
    if violations:
        details = "\n".join(
            f"  {name}: {path}" for name, path in violations
        )
        raise RuntimeError(
            "strict E2E may not consume frozen winning artifacts:\n" + details
        )


def preflight(
    pipeline: Pipeline,
    allow_commit_mismatch: bool,
    min_free_gib: float,
    check_runtime: bool,
) -> dict[str, object]:
    if check_runtime and sys.version_info[:2] != (3, 11):
        raise RuntimeError(
            "the pinned challenge runtime requires Python 3.11; "
            f"got {platform.python_version()}"
        )
    root = pipeline.challenge_root
    required = [
        root / "public_test_video_names.txt",
        root / "frame_utils.py",
        root / "modules.py",
        root / "evaluate.sh",
        root / "evaluate.py",
        root / "models" / "posenet.safetensors",
        root / "models" / "segnet.safetensors",
    ]
    if (root / "public_test_video_names.txt").is_file():
        required.extend(
            root / "videos" / name
            for name in (root / "public_test_video_names.txt").read_text().splitlines()
            if name
        )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "challenge checkout is incomplete:\n"
            + "\n".join(f"  {path}" for path in missing)
        )
    lfs_pointers = [
        path for path in required
        if path.suffix in {".mkv", ".safetensors"} and path.stat().st_size < 1_000_000
    ]
    if lfs_pointers:
        raise RuntimeError(
            "Git LFS assets are not materialized:\n"
            + "\n".join(f"  {path}" for path in lfs_pointers)
        )
    commit = git_commit(root)
    if commit != PINNED_CHALLENGE_COMMIT and not allow_commit_mismatch:
        raise RuntimeError(
            "challenge commit mismatch: "
            f"{commit} != {PINNED_CHALLENGE_COMMIT}; "
            "use the pinned checkout or pass --allow-challenge-commit-mismatch"
        )
    dirty = subprocess.run(
        [
            "git", "-C", str(root), "status",
            "--porcelain", "--untracked-files=no",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError(
            "challenge checkout has tracked modifications; use a clean pinned checkout"
        )
    pipeline.run_dir.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(pipeline.run_dir).free
    required_free = int(min_free_gib * (1024 ** 3))
    if free_bytes < required_free:
        raise RuntimeError(
            f"only {free_bytes / (1024 ** 3):.1f} GiB free; "
            f"{min_free_gib:.1f} GiB is required"
        )
    if check_runtime:
        dependency_check = (
            "import constriction, numpy, safetensors, torch; "
            "import nvidia.dali; "
            "assert torch.cuda.is_available(), 'CUDA is required'; "
            "print(torch.__version__, torch.cuda.get_device_name(0))"
        )
        subprocess.run([sys.executable, "-c", dependency_check], check=True)
    metadata = {
        "schema_version": 1,
        "created_at_unix": time.time(),
        "repository": str(ROOT),
        "challenge_root": str(root),
        "challenge_commit": commit,
        "expected_challenge_commit": PINNED_CHALLENGE_COMMIT,
        "commit_mismatch_allowed": allow_commit_mismatch,
        "python": sys.version,
        "platform": platform.platform(),
        "device": pipeline.device,
        "free_gib": free_bytes / (1024 ** 3),
        "runtime_identity": runtime_identity(),
    }
    (pipeline.run_dir / "run-meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return metadata


def select_stages(
    stages: list[Stage],
    from_stage: str | None,
    through_stage: str | None,
    skip_official_eval: bool,
) -> list[Stage]:
    names = [stage.name for stage in stages]
    start = 0 if from_stage is None else names.index(from_stage)
    end = len(stages) if through_stage is None else names.index(through_stage) + 1
    selected = stages[start:end]
    if skip_official_eval:
        selected = [
            stage for stage in selected
            if stage.name not in {
                "48_official_evaluation",
                "49_official_report",
            }
        ]
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Strict non-circular raw-video-to-CPR1 runner. Generated state is "
            "written only below --run-dir."
        )
    )
    parser.add_argument("action", choices=("plan", "run", "status"))
    parser.add_argument("--challenge-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "work" / "e2e")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--from-stage")
    parser.add_argument("--through-stage")
    parser.add_argument("--skip-official-eval", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-challenge-commit-mismatch", action="store_true")
    parser.add_argument("--min-free-gib", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = Pipeline(args.challenge_root, args.run_dir, args.device)
    frozen_guard(pipeline.stages)
    names = {stage.name for stage in pipeline.stages}
    for value, label in (
        (args.from_stage, "--from-stage"),
        (args.through_stage, "--through-stage"),
    ):
        if value is not None and value not in names:
            raise ValueError(f"{label} is not a known stage: {value}")
    selected = select_stages(
        pipeline.stages,
        args.from_stage,
        args.through_stage,
        args.skip_official_eval,
    )
    runner = Runner(pipeline, args.force)

    if args.action == "status":
        for stage in pipeline.stages:
            state = "complete" if runner.marker_is_valid(stage) else "pending"
            print(f"{state:8} {stage.name}  {stage.description}")
        return

    metadata = preflight(
        pipeline,
        args.allow_challenge_commit_mismatch,
        args.min_free_gib,
        check_runtime=args.action == "run",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    if args.action == "plan":
        for stage in selected:
            print(f"\n{stage.name}: {stage.description}")
            print(f"  {shlex.join(stage.command)}")
            print("  outputs:")
            for output in stage.outputs:
                print(f"    {output}")
        print(f"\n{len(selected)} stages; no commands executed")
        return

    for stage in selected:
        runner.run_stage(stage)
    runner.write_manifest(selected)
    print(
        "\nE2E complete. Final outputs:\n"
        f"  archive: {pipeline.run_dir / 'submission' / 'archive.zip'}\n"
        f"  report:  {pipeline.run_dir / 'submission' / 'report.txt'}\n"
        f"  metrics: {pipeline.report_dir / '49_official_metrics.json'}\n"
        f"  manifest:{pipeline.run_dir / 'e2e-manifest.json'}"
    )


if __name__ == "__main__":
    main()
