#!/usr/bin/env python3
"""Jacobian-guided joint integer search over deployed carrier coefficients."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from learned_pose_carrier_oracle import predict
from pose_basis_oracle import summarize
from search_pose_coeff_cpu import evaluate_all, score_candidates


N = 600


def proposal(
    posenet,
    master,
    target,
    basis,
    current,
    scales,
    amplitude,
    damping,
    max_step,
    active_dimensions,
):
    codes = current.float().detach().clone().requires_grad_(True)
    output = predict(
        posenet,
        master.float(),
        (codes * scales)[None],
        basis,
        amplitude,
        "gray",
    )[0]
    rows = []
    for dimension in range(output.numel()):
        rows.append(torch.autograd.grad(
            output[dimension], codes, retain_graph=dimension + 1 < output.numel()
        )[0])
    jacobian = torch.stack(rows).double()
    residual = (target[0] - output.detach()).double()
    gram = jacobian @ jacobian.T
    ridge = damping * max(float(gram.diag().mean()), 1e-12)
    try:
        dual = torch.linalg.solve(
            gram + ridge * torch.eye(gram.shape[0], device=gram.device), residual
        )
    except torch.linalg.LinAlgError:
        dual = torch.linalg.pinv(gram + ridge * torch.eye(
            gram.shape[0], device=gram.device
        )) @ residual
    update = (jacobian.T @ dual).clamp(-max_step, max_step).float()
    importance = update.abs() + 1e-3 * jacobian.square().sum(0).sqrt().float()
    active = importance.argsort(descending=True)[:active_dimensions]

    candidates = [current]
    for fraction in (0.25, 0.5, 1.0, 1.5):
        center = (current.float() + fraction * update).round().clamp(-2047, 2047)
        candidates.append(center.to(torch.int16))
    center = candidates[-2]
    for offsets in itertools.product((-1, 0, 1), repeat=active_dimensions):
        candidate = center.clone()
        for dimension, offset in zip(active, offsets, strict=True):
            candidate[dimension] = torch.clamp(
                candidate[dimension] + offset, -2047, 2047
            )
        candidates.append(candidate)
    return torch.unique(torch.stack(candidates), dim=0), {
        "jacobian_rank": int(torch.linalg.matrix_rank(jacobian)),
        "ridge": ridge,
        "update_norm": float(update.norm()),
        "active_dimensions": active.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge-root", type=Path, required=True)
    parser.add_argument("--target-cache", type=Path, required=True)
    parser.add_argument("--master-cache", type=Path, required=True)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--inner-iterations", type=int, default=3)
    parser.add_argument("--damping", type=float, default=0.01)
    parser.add_argument("--max-step", type=float, default=64.0)
    parser.add_argument("--active-dimensions", type=int, default=3)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--candidate-batch-size", type=int, default=32)
    parser.add_argument("--amplitude", type=float, default=64.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--save", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.active_dimensions <= 6:
        raise ValueError("--active-dimensions must be in [1,6]")

    device = torch.device(args.device)
    sys.path.insert(0, str(args.challenge_root.resolve()))
    import modules  # pylint: disable=import-error,import-outside-toplevel

    targets = torch.load(
        args.target_cache, map_location="cpu", weights_only=False
    )["pose"].float()
    master_payload = torch.load(
        args.master_cache, map_location="cpu", weights_only=False
    )
    masters = master_payload.get("masters", master_payload.get("frames"))
    initial = torch.load(args.init, map_location="cpu", weights_only=False)
    required = ("basis", "coeff_codes", "coeff_scales")
    if not all(key in initial for key in required):
        raise ValueError("joint search requires exact deployed carrier state")
    basis = initial["basis"].float().to(device)
    codes = initial["coeff_codes"].to(device=device, dtype=torch.int16)
    initial_codes = initial.get("initial_coeff_codes", initial["coeff_codes"]).cpu()
    scales = initial["coeff_scales"].float().to(device)

    posenet = modules.PoseNet().eval().to(device)
    posenet.load_state_dict(load_file(modules.posenet_sd_path, device=str(device)))
    for parameter in posenet.parameters():
        parameter.requires_grad_(False)

    history = []
    for pass_index in range(args.passes + 1):
        coeff = codes.float() * scales[None]
        errors = evaluate_all(
            posenet, masters, targets, basis, coeff, args.eval_batch_size,
            device, args.amplitude,
        )
        summary = summarize(errors, 3e-5)
        if pass_index == 0:
            print(json.dumps({"pass": 0, **summary}), flush=True)
            continue
        selected = errors.argsort(descending=True)[:args.top_k].tolist()
        accepted_rows = 0
        accepted_moves = 0
        improvement = 0.0
        for rank, pair_id in enumerate(selected, 1):
            current = codes[pair_id].clone()
            current_error = float(errors[pair_id])
            row_moves = []
            master = masters[pair_id:pair_id + 1].to(
                device=device, dtype=torch.float32
            )
            target = targets[pair_id:pair_id + 1].to(device)
            for iteration in range(args.inner_iterations):
                candidates, diagnostics = proposal(
                    posenet, master, target, basis, current, scales,
                    args.amplitude, args.damping, args.max_step,
                    args.active_dimensions,
                )
                candidate_errors = score_candidates(
                    posenet, master, target, basis, candidates, scales,
                    args.candidate_batch_size, device, args.amplitude,
                )
                winner = int(candidate_errors.argmin())
                winning_error = float(candidate_errors[winner])
                if winning_error >= current_error:
                    break
                row_moves.append({
                    "iteration": iteration,
                    "before": current_error,
                    "after": winning_error,
                    **diagnostics,
                })
                current = candidates[winner]
                current_error = winning_error
            if row_moves:
                improvement += float(errors[pair_id]) - current_error
                codes[pair_id] = current
                accepted_rows += 1
                accepted_moves += len(row_moves)
            print(json.dumps({
                "pass": pass_index,
                "rank": rank,
                "pair": pair_id,
                "before": float(errors[pair_id]),
                "after": current_error,
                "moves": len(row_moves),
            }), flush=True)

        coeff = codes.float() * scales[None]
        errors = evaluate_all(
            posenet, masters, targets, basis, coeff, args.eval_batch_size,
            device, args.amplitude,
        )
        record = {
            "pass": pass_index,
            "accepted_rows": accepted_rows,
            "accepted_moves": accepted_moves,
            "measured_improvement": improvement,
            **summarize(errors, 3e-5),
        }
        history.append(record)
        print(json.dumps(record), flush=True)

    result = {
        "quantized_basis_coeff": summarize(errors, 3e-5),
        "per_pair_mse": errors.tolist(),
        "history": history,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    args.save.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "basis": basis.detach().cpu(),
        "coeff": (codes.float() * scales[None]).detach().cpu(),
        "coeff_codes": codes.detach().cpu(),
        "coeff_scales": scales.detach().cpu(),
        "initial_coeff_codes": initial_codes,
        "result": result,
    }, args.save)


if __name__ == "__main__":
    main()
