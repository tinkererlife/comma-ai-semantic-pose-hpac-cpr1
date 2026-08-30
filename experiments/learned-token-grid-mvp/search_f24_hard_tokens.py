#!/usr/bin/env python3
"""Backprop-rank hard token flips on the exact public F24S renderer rail."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from learned_token_mvp import (
    EVAL_H,
    EVAL_W,
    N_TOKENS,
    N_TOTAL_PAIRS,
    camera_and_seg_input,
    evaluate_exact,
    expected_flip_loss,
    local_logits_from_tokens,
    official_metric_predictions,
    render_frozen_slaves,
    renderer_from_assignments,
    straight_through_one_hot,
)
from hpac_token_search import quantized_probability_bits


ORIGINAL_UNCOMPRESSED_BYTES = 37_545_489


class F24RateOracle:
    """Exact quantized ideal bits for F24S current/next-frame token effects."""

    def __init__(self, parts, renderer, runtime_root: Path, device: torch.device):
        from runtime.hpac_inference import optimize_sparse_evaluator
        from runtime.ihs2 import materialize_ihs1
        from runtime.residual_archive import _boundary_buckets, _sparse_class

        base_hpac = materialize_ihs1(parts.hpac_blob, renderer)
        self.model = renderer.load_hpac(base_hpac, device)
        self.sparse_class = _sparse_class(runtime_root / "cpr1")
        self.optimize_sparse_evaluator = optimize_sparse_evaluator
        self.eval_h = renderer.EVAL_H
        self.eval_w = renderer.EVAL_W
        self.sparse_by_batch = {}
        self.sparse = self._sparse(1)
        self.masks = renderer.group_masks(device)
        self.plans = []
        for mask in self.masks:
            positions = np.flatnonzero(mask.detach().cpu().numpy().reshape(-1))
            self.plans.append(torch.from_numpy(positions).to(device))
        self.table = torch.from_numpy(
            np.ascontiguousarray(parts.table.values, dtype=np.float32)
        ).to(device)
        self.boundary_buckets = _boundary_buckets
        self.device = device

    def _sparse(self, batch: int):
        sparse = self.sparse_by_batch.get(batch)
        if sparse is None:
            # Treat a batch as one vertically stacked image.  This preserves
            # the public sparse evaluator's fixed patch plans while matching
            # the contiguous [batch, H, W] memory layout exactly.
            sparse = self.sparse_class(
                self.model, self.eval_h * batch, self.eval_w
            )
            self.optimize_sparse_evaluator(sparse)
            self.sparse_by_batch[batch] = sparse
        return sparse

    @torch.no_grad()
    def frame_bits_batch(
        self,
        indices: list[int],
        targets: torch.Tensor,
        previous: torch.Tensor,
    ) -> torch.Tensor:
        sparse = self._sparse(len(indices))
        targets_device = targets.to(self.device).long()
        previous_device = previous.to(self.device).long()
        current = torch.zeros_like(previous_device)
        frame_ids = torch.tensor(indices, dtype=torch.long, device=self.device)
        context = self.model.prepare_frame_context(frame_ids, previous_device)
        boundary = torch.from_numpy(np.stack([
            self.boundary_buckets(frame.numpy()).reshape(-1)
            for frame in previous.to(torch.uint8).cpu()
        ])).to(self.device).long()
        totals = torch.zeros(len(indices), dtype=torch.float64, device=self.device)
        for group, (mask, positions) in enumerate(zip(self.masks, self.plans)):
            selected = sparse.selected_logits(current, context, group)
            predicted = selected.argmax(dim=1)
            feature = (
                boundary.index_select(1, positions).reshape(-1) * N_TOKENS
                + predicted
            )
            corrected = selected + self.table.index_select(0, feature)
            symbol_bits = quantized_probability_bits(corrected)
            symbols = targets_device[:, mask].reshape(-1)
            chosen = symbol_bits.gather(-1, symbols[:, None]).squeeze(-1)
            totals += chosen.reshape(len(indices), -1).double().sum(1)
            current[:, mask] = targets_device[:, mask]
        return totals.cpu()

    @torch.no_grad()
    def move_deltas(
        self,
        tokens: torch.Tensor,
        frame: int,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        count = len(candidates)
        previous = (
            torch.zeros_like(tokens[0]) if frame == 0 else tokens[frame - 1]
        )
        baseline = self.frame_bits_batch(
            [frame], tokens[frame:frame + 1], previous[None]
        )[0]
        proposed = self.frame_bits_batch(
            [frame] * count,
            candidates,
            previous[None].expand(count, -1, -1),
        )
        if frame + 1 < N_TOTAL_PAIRS:
            next_target = tokens[frame + 1:frame + 2]
            baseline += self.frame_bits_batch(
                [frame + 1], next_target, tokens[frame:frame + 1]
            )[0]
            proposed += self.frame_bits_batch(
                [frame + 1] * count,
                next_target.expand(count, -1, -1),
                candidates,
            )
        return proposed - baseline

    @torch.no_grad()
    def direct_symbol_bits(
        self,
        tokens: torch.Tensor,
        frame: int,
    ) -> torch.Tensor:
        """Return teacher-forced direct bit costs for every token category.

        This is a first-order proposal oracle only.  The exact move gate still
        recomputes the changed frame and its successor, so autoregressive
        downstream effects cannot create a false acceptance.
        """
        previous = (
            torch.zeros_like(tokens[0]) if frame == 0 else tokens[frame - 1]
        )
        target = tokens[frame:frame + 1].to(self.device).long()
        previous_device = previous[None].to(self.device).long()
        current = torch.zeros_like(previous_device)
        context = self.model.prepare_frame_context(
            torch.tensor([frame], dtype=torch.long, device=self.device),
            previous_device,
        )
        boundary = torch.from_numpy(
            self.boundary_buckets(previous.to(torch.uint8).cpu().numpy())
        ).to(self.device).long().reshape(1, -1)
        costs = torch.empty(
            1,
            self.eval_h,
            self.eval_w,
            N_TOKENS,
            dtype=torch.float32,
            device=self.device,
        )
        sparse = self._sparse(1)
        for group, (mask, positions) in enumerate(zip(self.masks, self.plans)):
            selected = sparse.selected_logits(current, context, group)
            predicted = selected.argmax(dim=1)
            feature = (
                boundary.index_select(1, positions).reshape(-1) * N_TOKENS
                + predicted
            )
            corrected = selected + self.table.index_select(0, feature)
            symbol_bits = quantized_probability_bits(corrected)
            costs[:, mask] = symbol_bits.reshape(1, -1, N_TOKENS)
            current[:, mask] = target[:, mask]
        return costs[0].cpu()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-runtime", type=Path, required=True)
    parser.add_argument("--challenge-root", type=Path, required=True)
    parser.add_argument("--target-cache", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--candidates-per-frame", type=int, default=8)
    parser.add_argument("--sweeps", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--init-margin", type=float, default=0.25)
    parser.add_argument("--exclude-pairs", type=int, nargs="*", default=[])
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-tokens", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    args = parser.parse_args()
    if args.top_k < 1 or args.candidates_per_frame < 1 or args.sweeps < 1:
        raise ValueError("search sizes must be positive")
    if any(pair < 0 or pair >= N_TOTAL_PAIRS for pair in args.exclude_pairs):
        raise ValueError("--exclude-pairs values must be in [0, 599]")
    return args


def load_renderer(path: Path):
    spec = importlib.util.spec_from_file_location("_f24_token_renderer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load renderer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_f24_state(args: argparse.Namespace, device: torch.device):
    runtime_root = args.submission_runtime.resolve()
    sys.path.insert(0, str(runtime_root))
    sys.path.insert(0, str(runtime_root / "cpr1"))
    from runtime.carrier_repack import (
        materialize_cpr1,
        split_frame0_selector_carrier,
    )
    from runtime.entropy.renderer_weight_codec import decode_wans1
    from runtime.hpac_inference import configure_cuda_reproducibility
    from runtime.residual_archive import read_residual_archive

    configure_cuda_reproducibility()
    renderer = load_renderer(runtime_root / "cpr1" / "inflate.py")
    parts = read_residual_archive(args.archive)
    packed_carrier, selector = split_frame0_selector_carrier(parts.carrier_blob)
    canonical = materialize_cpr1(packed_carrier, renderer)
    marker = bytes(40_252)
    semantic_pose = (
        struct.pack("<II", len(marker), len(canonical)) + marker + canonical
    )
    _, basis, coeff = renderer.unpack_semantic_pose(semantic_pose)
    semantic = renderer.SemanticTokenRenderer(96)
    records = decode_wans1(parts.semantic_blob)
    semantic.load_state_dict({
        record.schema.name: torch.from_numpy(
            np.ascontiguousarray(record.values, dtype=np.float32)
        )
        for record in records
    }, strict=True)
    for parameter in semantic.parameters():
        parameter.requires_grad_(False)
    rate_oracle = F24RateOracle(parts, renderer, runtime_root, device)
    return (
        semantic.eval().to(device),
        basis.float(),
        coeff.float(),
        selector,
        rate_oracle,
    )


def apply_selector(slaves: torch.Tensor, selector: bytes | None, runtime: Path) -> None:
    if selector is None:
        return
    sys.path.insert(0, str(runtime.resolve()))
    from runtime.frame0_selector import apply_pixel_mode, decode_selector

    modes, choices = decode_selector(selector)
    for mode_index, mode in enumerate(modes):
        frame_ids = np.flatnonzero(choices == mode_index)
        if not frame_ids.size:
            continue
        batch = slaves[frame_ids].permute(0, 2, 3, 1).numpy().copy()
        changed = apply_pixel_mode(batch, mode)
        slaves[frame_ids] = torch.from_numpy(changed).permute(0, 3, 1, 2)


def load_metric_models(challenge_root: Path, device: torch.device):
    sys.path.insert(0, str(challenge_root.resolve()))
    import modules

    segnet = modules.SegNet().eval().to(device)
    posenet = modules.PoseNet().eval().to(device)
    segnet.load_state_dict(load_file(modules.segnet_sd_path, device=str(device)))
    posenet.load_state_dict(load_file(modules.posenet_sd_path, device=str(device)))
    for model in (segnet, posenet):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return segnet, posenet


def candidate_moves(
    gradient: torch.Tensor,
    tokens: torch.Tensor,
    count: int,
    direct_rate_bits: torch.Tensor | None = None,
) -> list[tuple[int, int, int, int, float]]:
    current = tokens.to(gradient.device).long()
    current_gradient = gradient.gather(-1, current[..., None]).squeeze(-1)
    benefit = current_gradient[..., None] - gradient
    if direct_rate_bits is not None:
        current_bits = direct_rate_bits.gather(
            -1, current.cpu()[..., None]
        ).squeeze(-1)
        rate_benefit = (
            current_bits[..., None] - direct_rate_bits
        ).to(benefit.device)
        benefit = benefit + (
            25.0 * rate_benefit / (8.0 * ORIGINAL_UNCOMPRESSED_BYTES)
        )
    benefit.scatter_(-1, current[..., None], -torch.inf)
    flat = benefit.reshape(-1)
    keep = min(count, int(torch.isfinite(flat).sum()))
    values, indices = flat.topk(keep)
    moves = []
    for value, index in zip(values.tolist(), indices.tolist()):
        pixel, category = divmod(index, N_TOKENS)
        row, col = divmod(pixel, EVAL_W)
        moves.append((row, col, int(current[row, col]), category, value))
    return moves


@torch.no_grad()
def exact_candidates(
    semantic,
    segnet,
    posenet,
    token_grid: torch.Tensor,
    frame: int,
    moves: list[tuple[int, int, int, int, float]],
    slave: torch.Tensor,
    seg_target: torch.Tensor,
    pose_target: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    candidates = token_grid[None].repeat(len(moves), 1, 1)
    for index, (row, col, _, after, _) in enumerate(moves):
        candidates[index, row, col] = after
    ids = torch.full((len(moves),), frame, dtype=torch.long, device=device)
    masters = semantic(candidates.to(device).long(), ids)
    master_camera, _ = camera_and_seg_input(masters)
    seg_logits, pose = official_metric_predictions(
        segnet,
        posenet,
        slave.to(device).float().expand(len(moves), -1, -1, -1),
        master_camera,
    )
    mismatches = (
        seg_logits.argmax(1) != seg_target.to(device)[None]
    ).reshape(len(moves), -1).sum(1).cpu()
    pose_mse = (
        pose - pose_target.to(device)[None]
    ).square().mean(1).cpu()
    return candidates, mismatches, pose_mse


def score(seg_mismatches: float, pose_sum: float) -> float:
    seg = seg_mismatches / (N_TOTAL_PAIRS * EVAL_H * EVAL_W)
    pose = pose_sum / N_TOTAL_PAIRS
    return 100.0 * seg + math.sqrt(10.0 * pose)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    semantic, basis, coeff, selector, rate_oracle = load_f24_state(args, device)
    segnet, posenet = load_metric_models(args.challenge_root, device)
    targets = torch.load(
        args.target_cache, map_location="cpu", weights_only=False
    )
    raw = args.tokens.read_bytes()
    expected = N_TOTAL_PAIRS * EVAL_H * EVAL_W
    if len(raw) != expected:
        raise ValueError(f"token grid has {len(raw)} bytes, expected {expected}")
    tokens = torch.from_numpy(
        np.frombuffer(raw, dtype=np.uint8).copy().reshape(
            N_TOTAL_PAIRS, EVAL_H, EVAL_W
        )
    )
    pair_ids = list(range(N_TOTAL_PAIRS))
    slaves = render_frozen_slaves(basis, coeff, pair_ids, device)
    apply_selector(slaves, selector, args.submission_runtime)
    baseline = evaluate_exact(
        semantic,
        tokens,
        pair_ids,
        slaves,
        targets["seg"].long(),
        targets["pose"].float(),
        segnet,
        posenet,
        None,
        args.eval_batch_size,
        device,
        return_per_frame=True,
    )
    per_frame = baseline.pop("per_frame")
    seg_total = float(round(
        float(baseline["segnet_distortion"])
        * N_TOTAL_PAIRS * EVAL_H * EVAL_W
    ))
    pose_sum = float(baseline["posenet_distortion"]) * N_TOTAL_PAIRS
    current_score = score(seg_total, pose_sum)
    rate_delta_bits = 0.0
    print(json.dumps({
        "baseline_score_without_rate": current_score,
        **baseline,
    }), flush=True)
    history = []
    excluded = set(args.exclude_pairs)

    for sweep in range(1, args.sweeps + 1):
        ranking = sorted(
            (frame for frame in pair_ids if frame not in excluded),
            key=lambda frame: float(per_frame[frame]["posenet_distortion"]),
            reverse=True,
        )[:args.top_k]
        accepted = 0
        for rank, frame in enumerate(ranking, 1):
            local_tokens = tokens[frame]
            logits = local_logits_from_tokens(
                local_tokens[None], args.init_margin, device
            )
            assignments, _ = straight_through_one_hot(logits, args.temperature)
            master_eval = renderer_from_assignments(
                semantic,
                assignments,
                torch.tensor([frame], dtype=torch.long, device=device),
            )
            master_camera, _ = camera_and_seg_input(master_eval)
            seg_logits, pose = official_metric_predictions(
                segnet,
                posenet,
                slaves[frame:frame + 1].to(device).float(),
                master_camera,
            )
            pose_global = max(pose_sum / N_TOTAL_PAIRS, 1e-12)
            loss = (
                (100.0 / N_TOTAL_PAIRS)
                * expected_flip_loss(
                    seg_logits,
                    targets["seg"][frame:frame + 1].to(device).long(),
                )
                + (5.0 / math.sqrt(10.0 * pose_global) / N_TOTAL_PAIRS)
                * (pose - targets["pose"][frame:frame + 1].to(device)).square().mean()
            )
            loss.backward()
            direct_rate_bits = rate_oracle.direct_symbol_bits(tokens, frame)
            moves = candidate_moves(
                logits.grad[0],
                local_tokens,
                args.candidates_per_frame,
                direct_rate_bits,
            )
            candidates, mismatches, poses = exact_candidates(
                semantic,
                segnet,
                posenet,
                local_tokens,
                frame,
                moves,
                slaves[frame:frame + 1],
                targets["seg"][frame],
                targets["pose"][frame],
                device,
            )
            old_seg = float(round(
                float(per_frame[frame]["segnet_distortion"]) * EVAL_H * EVAL_W
            ))
            old_pose = float(per_frame[frame]["posenet_distortion"])
            candidate_scores = torch.tensor([
                score(
                    seg_total - old_seg + float(mismatch),
                    pose_sum - old_pose + float(candidate_pose),
                )
                for mismatch, candidate_pose in zip(mismatches, poses)
            ])
            rate_deltas = rate_oracle.move_deltas(
                tokens, frame, candidates
            )
            candidate_objectives = candidate_scores + (
                25.0
                * (rate_delta_bits + rate_deltas)
                / (8.0 * ORIGINAL_UNCOMPRESSED_BYTES)
            )
            current_objective = current_score + (
                25.0
                * rate_delta_bits
                / (8.0 * ORIGINAL_UNCOMPRESSED_BYTES)
            )
            best = int(candidate_objectives.argmin())
            new_score = float(candidate_scores[best])
            new_objective = float(candidate_objectives[best])
            accepted_move = new_objective < current_objective
            record = {
                "sweep": sweep,
                "rank": rank,
                "frame": frame,
                "before_score_without_rate": current_score,
                "after_score_without_rate": new_score,
                "before_objective": current_objective,
                "after_objective": new_objective,
                "incremental_rate_bits": float(rate_deltas[best]),
                "accepted": accepted_move,
                "move": {
                    "row": moves[best][0],
                    "col": moves[best][1],
                    "before": moves[best][2],
                    "after": moves[best][3],
                    "proposal_benefit": moves[best][4],
                },
            }
            if accepted_move:
                tokens[frame] = candidates[best]
                seg_total = seg_total - old_seg + float(mismatches[best])
                pose_sum = pose_sum - old_pose + float(poses[best])
                current_score = new_score
                rate_delta_bits += float(rate_deltas[best])
                per_frame[frame] = {
                    "segnet_distortion": float(mismatches[best]) / (EVAL_H * EVAL_W),
                    "posenet_distortion": float(poses[best]),
                    "semantic_pose_score_without_rate": 0.0,
                }
                accepted += 1
            history.append(record)
            print(json.dumps(record), flush=True)
        print(json.dumps({
            "sweep": sweep,
            "accepted": accepted,
            "score_without_rate": current_score,
            "objective_delta": (
                current_score
                + 25.0 * rate_delta_bits / (8.0 * ORIGINAL_UNCOMPRESSED_BYTES)
                - float(baseline["semantic_pose_score_without_rate"])
            ),
            "rate_delta_bits": rate_delta_bits,
            "segnet_distortion": seg_total / (N_TOTAL_PAIRS * EVAL_H * EVAL_W),
            "posenet_distortion": pose_sum / N_TOTAL_PAIRS,
        }), flush=True)

    args.out_tokens.parent.mkdir(parents=True, exist_ok=True)
    args.out_tokens.write_bytes(tokens.numpy().astype(np.uint8).tobytes())
    result = {
        "baseline": baseline,
        "final": {
            "score_without_rate": current_score,
            "rate_delta_bits": rate_delta_bits,
            "projected_objective_delta": (
                current_score
                + 25.0 * rate_delta_bits / (8.0 * ORIGINAL_UNCOMPRESSED_BYTES)
                - float(baseline["semantic_pose_score_without_rate"])
            ),
            "segnet_distortion": seg_total / (N_TOTAL_PAIRS * EVAL_H * EVAL_W),
            "posenet_distortion": pose_sum / N_TOTAL_PAIRS,
            "changed_tokens": int((tokens.numpy() != np.frombuffer(raw, dtype=np.uint8).reshape(tokens.shape)).sum()),
        },
        "history": history,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"baseline": baseline, "final": result["final"]}, indent=2))


if __name__ == "__main__":
    main()
