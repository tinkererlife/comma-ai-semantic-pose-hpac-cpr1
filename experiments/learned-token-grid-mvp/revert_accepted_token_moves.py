#!/usr/bin/env python3
"""T4-native reverse gate for previously accepted two-token moves."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


ORIGINAL_UNCOMPRESSED_BYTES = 37_545_489


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe-root", type=Path, required=True)
    parser.add_argument("--submission-runtime", type=Path, required=True)
    parser.add_argument("--challenge-root", type=Path, required=True)
    parser.add_argument("--target-cache", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--sweeps", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-tokens", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    return parser.parse_args()


def global_score(
    seg_mismatches: float,
    pose_sum: float,
    *,
    pairs: int,
    height: int,
    width: int,
) -> float:
    seg = seg_mismatches / (pairs * height * width)
    pose = pose_sum / pairs
    return 100.0 * seg + math.sqrt(10.0 * pose)


def main() -> None:
    args = parse_args()
    experiment_dir = args.recipe_root / "experiments" / "learned-token-grid-mvp"
    sys.path.insert(0, str(experiment_dir.resolve()))
    from learned_token_mvp import (
        EVAL_H,
        EVAL_W,
        N_TOTAL_PAIRS,
        evaluate_exact,
        render_frozen_slaves,
    )
    from search_f24_hard_tokens import (
        apply_selector,
        exact_candidates,
        load_f24_state,
        load_metric_models,
    )

    device = torch.device(args.device)
    semantic, basis, coeff, selector, rate_oracle = load_f24_state(args, device)
    segnet, posenet = load_metric_models(args.challenge_root, device)
    targets = torch.load(args.target_cache, map_location="cpu", weights_only=False)

    raw = args.tokens.read_bytes()
    expected = N_TOTAL_PAIRS * EVAL_H * EVAL_W
    if len(raw) != expected:
        raise ValueError(f"token grid has {len(raw)} bytes, expected {expected}")
    tokens = torch.from_numpy(
        np.frombuffer(raw, dtype=np.uint8)
        .copy()
        .reshape(N_TOTAL_PAIRS, EVAL_H, EVAL_W)
    )
    history_payload = json.loads(args.history.read_text())
    accepted_history = [
        record for record in history_payload["history"] if record["accepted"]
    ]

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
        * N_TOTAL_PAIRS
        * EVAL_H
        * EVAL_W
    ))
    pose_sum = float(baseline["posenet_distortion"]) * N_TOTAL_PAIRS
    perception_score = global_score(
        seg_total,
        pose_sum,
        pairs=N_TOTAL_PAIRS,
        height=EVAL_H,
        width=EVAL_W,
    )
    rate_delta_bits = 0.0
    print(json.dumps({
        "accepted_history_moves": len(accepted_history),
        "baseline_score_without_rate": perception_score,
        **baseline,
    }), flush=True)

    remaining = list(reversed(accepted_history))
    decisions = []
    for sweep in range(1, args.sweeps + 1):
        reverted_this_sweep = 0
        still_present = []
        for rank, record in enumerate(remaining, 1):
            frame = int(record["frame"])
            local_tokens = tokens[frame]
            reverse_moves = []
            matches = True
            for move in record["moves"]:
                row = int(move["row"])
                col = int(move["col"])
                after = int(move["after"])
                before = int(move["before"])
                if int(local_tokens[row, col]) != after:
                    matches = False
                    break
                reverse_moves.append((row, col, after, before, 0.0))
            if not matches:
                decisions.append({
                    "sweep": sweep,
                    "rank": rank,
                    "frame": frame,
                    "accepted_revert": False,
                    "reason": "current token no longer matches recorded after value",
                })
                continue

            # Duplicate the candidate so the T4 uses the same batch shape as
            # the official batch-16 rail (the final partial batch has size 8).
            gate_batch = (
                args.eval_batch_size
                if frame < N_TOTAL_PAIRS - (N_TOTAL_PAIRS % args.eval_batch_size)
                else N_TOTAL_PAIRS % args.eval_batch_size
            )
            repeated_moves = [reverse_moves] * gate_batch
            candidates, mismatches, poses = exact_candidates(
                semantic,
                segnet,
                posenet,
                local_tokens,
                frame,
                repeated_moves,
                slaves[frame:frame + 1],
                targets["seg"][frame],
                targets["pose"][frame],
                device,
            )
            # The deployed RC64 encoder is teacher-forced one frame at a time.
            # Perception keeps the official batch shape above; rate therefore
            # uses one candidate so its sparse HPAC kernels match deployment.
            rate_deltas = rate_oracle.move_deltas(tokens, frame, candidates[:1])

            old_seg = float(round(
                float(per_frame[frame]["segnet_distortion"]) * EVAL_H * EVAL_W
            ))
            old_pose = float(per_frame[frame]["posenet_distortion"])
            candidate_seg_total = seg_total - old_seg + float(mismatches[0])
            candidate_pose_sum = pose_sum - old_pose + float(poses[0])
            candidate_perception = global_score(
                candidate_seg_total,
                candidate_pose_sum,
                pairs=N_TOTAL_PAIRS,
                height=EVAL_H,
                width=EVAL_W,
            )
            candidate_rate_delta = rate_delta_bits + float(rate_deltas[0])
            current_objective = perception_score + (
                25.0 * rate_delta_bits / (8.0 * ORIGINAL_UNCOMPRESSED_BYTES)
            )
            candidate_objective = candidate_perception + (
                25.0
                * candidate_rate_delta
                / (8.0 * ORIGINAL_UNCOMPRESSED_BYTES)
            )
            accepted_revert = candidate_objective < current_objective
            decision = {
                "sweep": sweep,
                "rank": rank,
                "frame": frame,
                "before_objective": current_objective,
                "after_objective": candidate_objective,
                "objective_delta": candidate_objective - current_objective,
                "incremental_rate_bits": float(rate_deltas[0]),
                "accepted_revert": accepted_revert,
                "moves": [
                    {
                        "row": move[0],
                        "col": move[1],
                        "from": move[2],
                        "to": move[3],
                    }
                    for move in reverse_moves
                ],
            }
            if accepted_revert:
                tokens[frame] = candidates[0]
                seg_total = candidate_seg_total
                pose_sum = candidate_pose_sum
                perception_score = candidate_perception
                rate_delta_bits = candidate_rate_delta
                per_frame[frame] = {
                    "segnet_distortion": float(mismatches[0]) / (EVAL_H * EVAL_W),
                    "posenet_distortion": float(poses[0]),
                    "semantic_pose_score_without_rate": 0.0,
                }
                reverted_this_sweep += 1
            else:
                still_present.append(record)
            decisions.append(decision)
            print(json.dumps(decision), flush=True)
        print(json.dumps({
            "sweep": sweep,
            "accepted_reverts": reverted_this_sweep,
            "remaining_moves": len(still_present),
            "score_without_rate": perception_score,
            "rate_delta_bits": rate_delta_bits,
        }), flush=True)
        remaining = still_present
        if reverted_this_sweep == 0:
            break

    args.out_tokens.parent.mkdir(parents=True, exist_ok=True)
    args.out_tokens.write_bytes(tokens.numpy().astype(np.uint8).tobytes())
    result = {
        "baseline": baseline,
        "final": {
            "score_without_rate": perception_score,
            "rate_delta_bits": rate_delta_bits,
            "projected_objective_delta": (
                perception_score
                + 25.0 * rate_delta_bits / (8.0 * ORIGINAL_UNCOMPRESSED_BYTES)
                - float(baseline["semantic_pose_score_without_rate"])
            ),
            "segnet_distortion": seg_total / (N_TOTAL_PAIRS * EVAL_H * EVAL_W),
            "posenet_distortion": pose_sum / N_TOTAL_PAIRS,
            "reverted_pairs": sum(d["accepted_revert"] for d in decisions),
            "changed_tokens": int((tokens.numpy() != np.frombuffer(raw, dtype=np.uint8).reshape(tokens.shape)).sum()),
        },
        "decisions": decisions,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["final"], indent=2), flush=True)


if __name__ == "__main__":
    main()
