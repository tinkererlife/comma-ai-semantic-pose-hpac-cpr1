"""Exact deployed-HPAC rate oracle and batched discrete search helpers."""

from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F


N_TOKENS = 5
N_TOTAL_PAIRS = 600
ORIGINAL_UNCOMPRESSED_BYTES = 37_545_489


@dataclass(frozen=True)
class TokenMove:
    """One hard categorical change proposed by the first-order objective."""

    batch_index: int
    parameter_index: int
    row: int
    col: int
    before: int
    after: int
    benefit: float
    perception_benefit: float
    direct_rate_benefit_bits: float


@dataclass
class BacktrackResult:
    tokens: torch.Tensor
    key: float
    payload: object
    accepted_moves: list[TokenMove]
    evaluations: int
    rejected_batches: int


def projected_hpac_rate_score(bits: float, frames: int) -> float:
    """Convert token bits to the official score scale for a frame window."""
    if frames < 1:
        raise ValueError("frames must be positive")
    return (
        25.0
        * bits
        * N_TOTAL_PAIRS
        / (8.0 * frames * ORIGINAL_UNCOMPRESSED_BYTES)
    )


def quantized_probability_bits(selected_logits: torch.Tensor) -> torch.Tensor:
    """Match codec_hpac_integer.probability_table and return per-symbol bits.

    The deployed range coder rounds logits to eighths, evaluates softmax in
    float64, then stores float32 probabilities.  Mirroring those operations
    makes this the same ideal-bit yardstick reported by the real encoder.
    """
    codes = (
        selected_logits.mul(8)
        .round()
        .clamp(-32768, 32767)
        .to(torch.int16)
        .cpu()
        .numpy()
    )
    logits = codes.astype(np.float64) / 8.0
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    probabilities = probabilities.astype(np.float32)
    bits = -np.log2(probabilities.astype(np.float64))
    return torch.from_numpy(bits.astype(np.float32, copy=False))


class HPACRateOracle:
    """Evaluate token surprise with the frozen, deployed #130 IntegerHPAC."""

    def __init__(
        self,
        checkpoint: Path,
        source_root: Path,
        baseline_tokens: torch.Tensor,
        start_pair: int,
        device: torch.device,
    ) -> None:
        source_root = source_root.resolve()
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
        from codec_hpac_integer import load_model
        from codec_hpac_residual import group_masks
        from hpac_integer_sparse import SparseIntegerHPAC

        config = SimpleNamespace(
            channels=64,
            patch=64,
            delta=2,
            frame_dim=8,
            norm_mode="none",
            activation="relu",
            frame_scale=True,
            weight_bound=127,
            activation_bound=127,
            weight_scales=True,
            weight_exponent_min=-6,
            spm=True,
            norm_gates=False,
            self_compress=True,
        )
        self.model = load_model(checkpoint, config, device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.sparse = SparseIntegerHPAC(self.model)
        self.masks = group_masks(config.patch, config.delta, device)
        self.cpu_masks = [mask.cpu() for mask in self.masks]
        self.baseline_tokens = baseline_tokens.to(torch.uint8).cpu()
        self.start_pair = start_pair
        self.device = device
        if self.baseline_tokens.ndim != 3:
            raise ValueError("baseline token tensor must have shape [frames, H, W]")
        if self.baseline_tokens.shape[0] != N_TOTAL_PAIRS:
            raise ValueError("the #130 HPAC oracle requires all 600 baseline frames")

    def _raw_frame(
        self,
        window_tokens: torch.Tensor,
        global_index: int,
        overrides: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        local_index = global_index - self.start_pair
        if 0 <= local_index < len(window_tokens):
            return overrides.get(local_index, window_tokens[local_index]).to(
                torch.uint8
            ).cpu()
        return self.baseline_tokens[global_index]

    @torch.no_grad()
    def frame_bits(
        self,
        global_index: int,
        target_raw: torch.Tensor,
        previous_raw: torch.Tensor,
        return_costs: bool = False,
    ) -> tuple[float, torch.Tensor | None]:
        target = target_raw.to(device=self.device, dtype=torch.long)
        previous = previous_raw.to(device=self.device, dtype=torch.long).view(
            1, *previous_raw.shape
        )
        current = torch.zeros_like(previous)
        index = torch.tensor([global_index], dtype=torch.long, device=self.device)
        context = self.model.prepare_frame_context(index, previous)
        costs = (
            torch.empty((*target_raw.shape, N_TOKENS), dtype=torch.float32)
            if return_costs
            else None
        )
        total_bits = 0.0
        for group, (mask, cpu_mask) in enumerate(zip(self.masks, self.cpu_masks)):
            selected = self.sparse.selected_logits(current, context, group)
            selected_bits = quantized_probability_bits(selected)
            symbols = target_raw[cpu_mask].long().cpu()
            rows = torch.arange(len(symbols))
            total_bits += float(selected_bits[rows, symbols].double().sum())
            if costs is not None:
                costs[cpu_mask] = selected_bits
            current[0, mask] = target[mask]
        return total_bits, costs

    @torch.no_grad()
    def affected_bits(
        self,
        window_tokens: torch.Tensor,
        local_indices: list[int],
        overrides: dict[int, torch.Tensor] | None = None,
        return_cost_tables: bool = False,
    ) -> tuple[float, dict[int, torch.Tensor]]:
        """Rate for changed frames plus their one-frame temporal dependents."""
        overrides = overrides or {}
        global_frames: set[int] = set()
        for local_index in local_indices:
            global_index = self.start_pair + local_index
            global_frames.add(global_index)
            if global_index + 1 < N_TOTAL_PAIRS:
                global_frames.add(global_index + 1)

        wanted_tables = set(local_indices) if return_cost_tables else set()
        tables: dict[int, torch.Tensor] = {}
        total_bits = 0.0
        zero = torch.zeros_like(self.baseline_tokens[0])
        for global_index in sorted(global_frames):
            target = self._raw_frame(window_tokens, global_index, overrides)
            previous = (
                zero
                if global_index == 0
                else self._raw_frame(window_tokens, global_index - 1, overrides)
            )
            local_index = global_index - self.start_pair
            bits, costs = self.frame_bits(
                global_index,
                target,
                previous,
                return_costs=local_index in wanted_tables,
            )
            total_bits += bits
            if costs is not None:
                tables[local_index] = costs
        return total_bits, tables


def rank_token_moves(
    logits: torch.Tensor,
    current_tokens: torch.Tensor,
    selected: list[int],
    max_pixels_per_frame: int,
    attempted_masks: list[torch.Tensor] | None = None,
    category_rate_bits: torch.Tensor | None = None,
    rate_score_per_bit: float = 0.0,
    minimum_distance: int = 0,
    rate_only: bool = False,
) -> tuple[list[TokenMove], dict[str, float | int | str]]:
    """Rank hard moves using perception gradients plus deployed-HPAC surprise."""
    if logits.grad is None:
        raise ValueError("logits must have a gradient before ranking moves")
    if max_pixels_per_frame < 1:
        raise ValueError("max_pixels_per_frame must be positive")
    if minimum_distance < 0:
        raise ValueError("minimum_distance cannot be negative")
    if logits.shape[:-1] != current_tokens.shape:
        raise ValueError("logits and token shapes do not match")
    if len(selected) != logits.shape[0]:
        raise ValueError("selected indices do not match the batch size")

    gradient = logits.grad.detach()
    current = current_tokens.to(logits.device).long()
    if category_rate_bits is None:
        rate_bits = torch.zeros_like(gradient)
    else:
        if category_rate_bits.shape != gradient.shape:
            raise ValueError("category rate table does not match logits")
        rate_bits = category_rate_bits.to(logits.device)

    perception_objective = torch.zeros_like(gradient) if rate_only else gradient
    objective = perception_objective + rate_score_per_bit * rate_bits
    current_objective = objective.gather(-1, current.unsqueeze(-1)).squeeze(-1)
    alternative_objective, alternative_token = objective.masked_fill(
        F.one_hot(current, N_TOKENS).bool(), torch.inf
    ).min(dim=-1)
    benefit = current_objective - alternative_objective

    current_perception = gradient.gather(-1, current.unsqueeze(-1)).squeeze(-1)
    alternative_perception = gradient.gather(
        -1, alternative_token.unsqueeze(-1)
    ).squeeze(-1)
    perception_benefit = current_perception - alternative_perception
    current_rate = rate_bits.gather(-1, current.unsqueeze(-1)).squeeze(-1)
    alternative_rate = rate_bits.gather(
        -1, alternative_token.unsqueeze(-1)
    ).squeeze(-1)
    rate_benefit = current_rate - alternative_rate

    moves: list[TokenMove] = []
    positive_candidates = 0
    for batch_index, parameter_index in enumerate(selected):
        flat_benefit = benefit[batch_index].reshape(-1)
        positive = flat_benefit > 0
        if attempted_masks is not None:
            attempted = attempted_masks[parameter_index].reshape(-1).to(logits.device)
            positive &= ~attempted
        positive_count = int(positive.sum())
        positive_candidates += positive_count
        if not positive_count:
            continue

        pool_size = min(positive_count, max_pixels_per_frame * 64)
        candidates = flat_benefit.masked_fill(~positive, -torch.inf)
        ranked = candidates.topk(pool_size, sorted=True).indices
        chosen_coordinates: list[tuple[int, int]] = []
        width = current_tokens.shape[2]
        for flat_index_tensor in ranked:
            flat_index = int(flat_index_tensor)
            row, col = divmod(flat_index, width)
            if minimum_distance and any(
                max(abs(row - other_row), abs(col - other_col))
                < minimum_distance
                for other_row, other_col in chosen_coordinates
            ):
                continue
            before = int(current_tokens[batch_index, row, col])
            after = int(alternative_token[batch_index, row, col])
            moves.append(TokenMove(
                batch_index=batch_index,
                parameter_index=parameter_index,
                row=row,
                col=col,
                before=before,
                after=after,
                benefit=float(benefit[batch_index, row, col]),
                perception_benefit=float(
                    perception_benefit[batch_index, row, col]
                ),
                direct_rate_benefit_bits=float(
                    rate_benefit[batch_index, row, col]
                ),
            ))
            chosen_coordinates.append((row, col))
            if attempted_masks is not None:
                attempted_masks[parameter_index][row, col] = True
            if len(chosen_coordinates) == max_pixels_per_frame:
                break

    return moves, {
        "gradient_selected_pixels": len(moves),
        "gradient_positive_candidates": positive_candidates,
        "gradient_largest_benefit": max(
            (move.benefit for move in moves), default=0.0
        ),
        "gradient_direct_rate_benefit_bits": sum(
            move.direct_rate_benefit_bits for move in moves
        ),
        "proposal_mode": "rate" if rate_only else "joint",
    }


def apply_token_moves(tokens: torch.Tensor, moves: list[TokenMove]) -> torch.Tensor:
    output = tokens.clone()
    for move in moves:
        if int(output[move.batch_index, move.row, move.col]) != move.before:
            raise ValueError("token move no longer matches its source state")
        output[move.batch_index, move.row, move.col] = move.after
    return output


def accept_with_backtracking(
    initial_tokens: torch.Tensor,
    moves: list[TokenMove],
    initial_key: float,
    initial_payload: object,
    evaluate: Callable[[torch.Tensor], tuple[float, object]],
    max_evaluations: int,
) -> BacktrackResult:
    """Accept a batch or recursively split it into independently useful moves."""
    if max_evaluations < 1:
        raise ValueError("max_evaluations must be positive")
    if not moves:
        return BacktrackResult(
            tokens=initial_tokens.clone(),
            key=initial_key,
            payload=initial_payload,
            accepted_moves=[],
            evaluations=0,
            rejected_batches=0,
        )

    working = initial_tokens.clone()
    key = initial_key
    payload = initial_payload
    accepted: list[TokenMove] = []
    evaluations = 0
    rejected_batches = 0
    queue: deque[list[TokenMove]] = deque([moves])
    while queue and evaluations < max_evaluations:
        subset = queue.popleft()
        candidate = apply_token_moves(working, subset)
        candidate_key, candidate_payload = evaluate(candidate)
        evaluations += 1
        if candidate_key < key:
            working = candidate
            key = candidate_key
            payload = candidate_payload
            accepted.extend(subset)
            continue

        rejected_batches += 1
        if len(subset) > 1 and evaluations < max_evaluations:
            middle = (len(subset) + 1) // 2
            queue.appendleft(subset[middle:])
            queue.appendleft(subset[:middle])

    return BacktrackResult(
        tokens=working,
        key=key,
        payload=payload,
        accepted_moves=accepted,
        evaluations=evaluations,
        rejected_batches=rejected_batches,
    )
