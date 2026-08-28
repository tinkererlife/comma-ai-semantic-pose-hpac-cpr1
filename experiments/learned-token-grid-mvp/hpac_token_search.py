"""Exact deployed-HPAC rate oracle and batched discrete search helpers."""

from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

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


@dataclass(frozen=True)
class TokenRegionMove:
    """Set one small rectangular token region to a shared category."""

    batch_index: int
    parameter_index: int
    row: int
    col: int
    height: int
    width: int
    after: int
    benefit: float
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
    codes = selected_logits.mul(8).round().clamp(-32768, 32767).to(torch.int16)
    logits = codes.to(torch.float64) / 8.0
    logits = logits - logits.amax(dim=1, keepdim=True)
    probabilities = logits.exp()
    probabilities = (probabilities / probabilities.sum(dim=1, keepdim=True)).float()
    return -probabilities.to(torch.float64).log2().float()


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
            torch.empty(
                (*target_raw.shape, N_TOKENS),
                dtype=torch.float32,
                device=self.device,
            )
            if return_costs
            else None
        )
        total_bits = torch.zeros((), dtype=torch.float64, device=self.device)
        for group, mask in enumerate(self.masks):
            selected = self.sparse.selected_logits(current, context, group)
            selected_bits = quantized_probability_bits(selected)
            symbols = target[mask]
            rows = torch.arange(len(symbols), device=self.device)
            total_bits += selected_bits[rows, symbols].double().sum()
            if costs is not None:
                costs[mask] = selected_bits
            current[0, mask] = target[mask]
        return float(total_bits), costs

    @torch.no_grad()
    def frame_bits_batch(
        self,
        global_indices: list[int],
        targets_raw: torch.Tensor,
        previous_raw: torch.Tensor,
        return_costs: bool = False,
    ) -> tuple[list[float], torch.Tensor | None]:
        """Vectorized full-frame HPAC evaluation for independent frames."""
        if not global_indices:
            return [], None
        targets = targets_raw.to(device=self.device, dtype=torch.long)
        previous = previous_raw.to(device=self.device, dtype=torch.long)
        current = torch.zeros_like(previous)
        indices = torch.tensor(
            global_indices, dtype=torch.long, device=self.device
        )
        context = self.model.prepare_frame_context(indices, previous)
        costs = (
            torch.empty(
                (*targets.shape, N_TOKENS),
                dtype=torch.float32,
                device=self.device,
            )
            if return_costs
            else None
        )
        totals = torch.zeros(
            len(global_indices), dtype=torch.float64, device=self.device
        )
        for group, mask in enumerate(self.masks):
            logits = self.sparse.selected_logits(current, context, group)
            selected_bits = quantized_probability_bits(logits).reshape(
                len(global_indices), -1, N_TOKENS
            )
            symbols = targets[:, mask]
            totals += selected_bits.gather(
                -1, symbols.unsqueeze(-1)
            ).squeeze(-1).double().sum(dim=1)
            if costs is not None:
                costs[:, mask] = selected_bits
            current[:, mask] = targets[:, mask]
        return totals.cpu().tolist(), costs

    @torch.no_grad()
    def proposal_batch(
        self,
        window_tokens: torch.Tensor,
        local_indices: list[int],
        cached_frame_bits: dict[int, float] | None = None,
    ) -> tuple[float, dict[int, torch.Tensor], dict[int, float]]:
        """Batched current-frame cost tables plus cached temporal dependents."""
        cached_frame_bits = cached_frame_bits or {}
        global_indices = [self.start_pair + index for index in local_indices]
        zero = torch.zeros_like(self.baseline_tokens[0])
        targets = torch.stack([
            self._raw_frame(window_tokens, index, {}) for index in global_indices
        ])
        previous = torch.stack([
            zero
            if index == 0
            else self._raw_frame(window_tokens, index - 1, {})
            for index in global_indices
        ])
        current_totals, costs = self.frame_bits_batch(
            global_indices, targets, previous, return_costs=True
        )
        frame_totals = dict(zip(global_indices, current_totals))

        missing_next = sorted({
            index + 1
            for index in global_indices
            if index + 1 < N_TOTAL_PAIRS and index + 1 not in cached_frame_bits
        })
        if missing_next:
            next_targets = torch.stack([
                self._raw_frame(window_tokens, index, {}) for index in missing_next
            ])
            next_previous = torch.stack([
                self._raw_frame(window_tokens, index - 1, {})
                for index in missing_next
            ])
            next_totals, _ = self.frame_bits_batch(
                missing_next, next_targets, next_previous
            )
            frame_totals.update(zip(missing_next, next_totals))
        for index in global_indices:
            next_index = index + 1
            if next_index < N_TOTAL_PAIRS and next_index not in frame_totals:
                frame_totals[next_index] = cached_frame_bits[next_index]

        tables = {
            local_index: costs[batch_index]
            for batch_index, local_index in enumerate(local_indices)
        }
        affected = set(global_indices)
        affected.update(
            index + 1 for index in global_indices if index + 1 < N_TOTAL_PAIRS
        )
        selected_totals = {
            index: frame_totals[index] for index in affected
        }
        return sum(selected_totals.values()), tables, selected_totals

    def _patch_tokens(self, raw: torch.Tensor) -> torch.Tensor:
        patch = self.model.P
        height, width = raw.shape
        return (
            raw.view(height // patch, patch, width // patch, patch)
            .permute(0, 2, 1, 3)
            .reshape(-1, patch, patch)
        )

    @torch.no_grad()
    def patch_bits(
        self,
        global_index: int,
        target_raw: torch.Tensor,
        previous_raw: torch.Tensor,
        patch_indices: torch.Tensor,
        context=None,
        groups: set[int] | None = None,
    ) -> float:
        """Exact ideal bits for selected independent 64x64 HPAC patches."""
        patch_indices = torch.as_tensor(
            patch_indices, dtype=torch.long, device=self.device
        ).unique(sorted=True)
        if not len(patch_indices):
            return 0.0
        target = target_raw.to(device=self.device, dtype=torch.long)
        previous = previous_raw.to(device=self.device, dtype=torch.long).view(
            1, *previous_raw.shape
        )
        current = torch.zeros_like(previous)
        if context is None:
            index = torch.tensor(
                [global_index], dtype=torch.long, device=self.device
            )
            context = self.model.prepare_frame_context(index, previous)
        target_patches = self._patch_tokens(target).index_select(0, patch_indices)
        total_bits = torch.zeros((), dtype=torch.float64, device=self.device)
        for group, mask in enumerate(self.masks):
            if groups is None or group in groups:
                logits = self.sparse.selected_logits_patches(
                    current, context, group, patch_indices
                )
                selected_bits = quantized_probability_bits(
                    logits.reshape(-1, N_TOKENS)
                )
                positions = self.sparse.plans[group].targets
                symbols = target_patches[
                    :, positions[:, 0], positions[:, 1]
                ].reshape(-1)
                rows = torch.arange(len(symbols), device=self.device)
                total_bits += selected_bits[rows, symbols].double().sum()
            current[0, mask] = target[mask]
        return float(total_bits)

    @torch.no_grad()
    def patch_bits_pairs(
        self,
        global_indices: list[int],
        targets_raw: torch.Tensor,
        previous_raw: torch.Tensor,
        frame_indices: list[int],
        patch_indices: list[int],
        group_sets: list[set[int]],
        context=None,
    ) -> list[float]:
        """Evaluate selected (frame, patch) pairs in group-vectorized batches."""
        if not frame_indices:
            return []
        if not (
            len(frame_indices) == len(patch_indices) == len(group_sets)
        ):
            raise ValueError("patch-pair metadata lengths do not match")
        target = targets_raw.to(device=self.device, dtype=torch.long)
        previous = previous_raw.to(device=self.device, dtype=torch.long)
        current = torch.zeros_like(previous)
        indices = torch.tensor(
            global_indices, dtype=torch.long, device=self.device
        )
        if context is None:
            context = self.model.prepare_frame_context(indices, previous)
        frame_tensor = torch.tensor(
            frame_indices, dtype=torch.long, device=self.device
        )
        patch_tensor = torch.tensor(
            patch_indices, dtype=torch.long, device=self.device
        )
        flat_indices = frame_tensor * self.sparse.patch_count + patch_tensor
        target_patches = target.view(
            target.shape[0],
            self.sparse.patch_rows,
            self.sparse.patch,
            self.sparse.patch_cols,
            self.sparse.patch,
        ).permute(0, 1, 3, 2, 4).reshape(
            -1, self.sparse.patch, self.sparse.patch
        ).index_select(0, flat_indices)
        totals = torch.zeros(
            len(frame_indices), dtype=torch.float64, device=self.device
        )
        for group, mask in enumerate(self.masks):
            active = [
                pair_index
                for pair_index, groups in enumerate(group_sets)
                if group in groups
            ]
            if active:
                active_tensor = torch.tensor(
                    active, dtype=torch.long, device=self.device
                )
                logits = self.sparse.selected_logits_patches(
                    current,
                    context,
                    group,
                    patch_tensor.index_select(0, active_tensor),
                    frame_tensor.index_select(0, active_tensor),
                )
                selected_bits = quantized_probability_bits(
                    logits.reshape(-1, N_TOKENS)
                ).reshape(len(active), -1, N_TOKENS)
                positions = self.sparse.plans[group].targets
                active_targets = target_patches.index_select(
                    0, active_tensor
                )
                symbols = active_targets[
                    :, positions[:, 0], positions[:, 1]
                ]
                contributions = selected_bits.gather(
                    -1, symbols.unsqueeze(-1)
                ).squeeze(-1).double().sum(dim=1)
                totals[active_tensor] += contributions
            current[:, mask] = target[:, mask]
        return totals.cpu().tolist()

    def _future_positions(
        self,
        positions: set[tuple[int, int]],
        offsets: list[tuple[int, int]],
    ) -> set[tuple[int, int]]:
        patch = self.model.P
        return {
            (row - dy, col - dx)
            for row, col in positions
            for dy, dx in offsets
            if 0 <= row - dy < patch and 0 <= col - dx < patch
        }

    def _output_groups(
        self,
        hidden_positions: set[tuple[int, int]],
    ) -> set[int]:
        positions = self._future_positions(
            hidden_positions, self.sparse.b1_offsets
        )
        positions = self._future_positions(positions, self.sparse.b2_offsets)
        return {col + self.model.delta * row for row, col in positions}

    @torch.no_grad()
    def localized_move_delta(
        self,
        window_tokens: torch.Tensor,
        local_index: int,
        candidate_raw: torch.Tensor,
    ) -> tuple[float, dict[int, float], dict[str, int]]:
        """Exact rate delta for hard edits in one frame using affected patches."""
        global_index = self.start_pair + local_index
        before_raw = self._raw_frame(window_tokens, global_index, {})
        candidate_raw = candidate_raw.to(torch.uint8).cpu()
        changed = (before_raw != candidate_raw).nonzero(as_tuple=False)
        if not len(changed):
            return 0.0, {}, {"current_patches": 0, "next_patches": 0}

        patch = self.model.P
        patch_cols = before_raw.shape[1] // patch
        current_patches = (
            (changed[:, 0] // patch) * patch_cols + changed[:, 1] // patch
        ).unique(sorted=True).to(self.device)
        local_changed = {
            (int(row) % patch, int(col) % patch)
            for row, col in changed.tolist()
        }
        current_hidden = self._future_positions(
            local_changed, self.sparse.a_offsets
        )
        current_groups = self._output_groups(current_hidden)
        current_groups |= {
            col + self.model.delta * row for row, col in local_changed
        }
        zero = torch.zeros_like(self.baseline_tokens[0])
        previous_raw = (
            zero
            if global_index == 0
            else self._raw_frame(window_tokens, global_index - 1, {})
        )
        index = torch.tensor(
            [global_index], dtype=torch.long, device=self.device
        )
        current_context = self.model.prepare_frame_context(
            index, previous_raw.to(self.device).long().unsqueeze(0)
        )
        before_current = self.patch_bits(
            global_index,
            before_raw,
            previous_raw,
            current_patches,
            context=current_context,
            groups=current_groups,
        )
        after_current = self.patch_bits(
            global_index,
            candidate_raw,
            previous_raw,
            current_patches,
            context=current_context,
            groups=current_groups,
        )
        frame_deltas = {global_index: after_current - before_current}

        next_patch_count = 0
        if global_index + 1 < N_TOTAL_PAIRS:
            next_raw = self._raw_frame(window_tokens, global_index + 1, {})
            index = torch.tensor(
                [global_index + 1], dtype=torch.long, device=self.device
            )
            before_previous = before_raw.to(self.device).long().unsqueeze(0)
            after_previous = candidate_raw.to(self.device).long().unsqueeze(0)
            before_context = self.model.prepare_frame_context(
                index, before_previous
            )
            after_context = self.model.prepare_frame_context(index, after_previous)
            affected = torch.zeros(
                self.sparse.patch_count, dtype=torch.bool, device=self.device
            )
            for before_value, after_value in zip(before_context, after_context):
                if before_value is None:
                    continue
                affected |= (before_value != after_value).reshape(
                    self.sparse.patch_count, -1
                ).any(dim=1)
            next_patches = affected.nonzero(as_tuple=False).flatten()
            next_patch_count = len(next_patches)
            context_positions: set[tuple[int, int]] = set()
            for before_value, after_value in zip(before_context, after_context):
                if before_value is None or before_value.shape[-2:] == (1, 1):
                    continue
                changed_context = (before_value != after_value).any(dim=1)
                for patch_map in changed_context.index_select(0, next_patches):
                    context_positions.update(
                        map(tuple, patch_map.nonzero(as_tuple=False).tolist())
                    )
            next_groups = self._output_groups(context_positions)
            before_next = self.patch_bits(
                global_index + 1,
                next_raw,
                before_raw,
                next_patches,
                context=before_context,
                groups=next_groups,
            )
            after_next = self.patch_bits(
                global_index + 1,
                next_raw,
                candidate_raw,
                next_patches,
                context=after_context,
                groups=next_groups,
            )
            frame_deltas[global_index + 1] = after_next - before_next

        return (
            sum(frame_deltas.values()),
            frame_deltas,
            {
                "current_patches": len(current_patches),
                "next_patches": next_patch_count,
                "current_groups": len(current_groups),
                "next_groups": len(next_groups) if next_patch_count else 0,
            },
        )

    @torch.no_grad()
    def localized_move_delta_batch(
        self,
        window_tokens: torch.Tensor,
        local_indices: list[int],
        candidate_raws: torch.Tensor,
    ) -> list[tuple[float, dict[int, float], dict[str, int]]]:
        """Vectorize exact localized deltas for independent one-frame moves."""
        if len(local_indices) != len(candidate_raws):
            raise ValueError("candidate batch does not match local indices")
        results: list[
            tuple[float, dict[int, float], dict[str, int]] | None
        ] = [None] * len(local_indices)
        entries: list[dict[str, object]] = []
        patch = self.model.P
        patch_cols = self.baseline_tokens.shape[2] // patch
        zero = torch.zeros_like(self.baseline_tokens[0])
        for result_index, (local_index, candidate_raw) in enumerate(
            zip(local_indices, candidate_raws)
        ):
            global_index = self.start_pair + local_index
            before_raw = self._raw_frame(window_tokens, global_index, {})
            candidate_raw = candidate_raw.to(torch.uint8).cpu()
            changed = (before_raw != candidate_raw).nonzero(as_tuple=False)
            if not len(changed):
                results[result_index] = (
                    0.0,
                    {},
                    {"current_patches": 0, "next_patches": 0},
                )
                continue
            current_patches = (
                (changed[:, 0] // patch) * patch_cols + changed[:, 1] // patch
            ).unique(sorted=True).tolist()
            local_changed = {
                (int(row) % patch, int(col) % patch)
                for row, col in changed.tolist()
            }
            current_hidden = self._future_positions(
                local_changed, self.sparse.a_offsets
            )
            current_groups = self._output_groups(current_hidden)
            current_groups |= {
                col + self.model.delta * row for row, col in local_changed
            }
            previous_raw = (
                zero
                if global_index == 0
                else self._raw_frame(window_tokens, global_index - 1, {})
            )
            entries.append({
                "result_index": result_index,
                "local_index": local_index,
                "global_index": global_index,
                "before": before_raw,
                "candidate": candidate_raw,
                "previous": previous_raw,
                "current_patches": current_patches,
                "current_groups": current_groups,
            })

        if not entries:
            return [result for result in results if result is not None]

        batch = len(entries)
        current_targets = torch.stack(
            [entry["before"] for entry in entries]
            + [entry["candidate"] for entry in entries]
        )
        current_previous = torch.stack(
            [entry["previous"] for entry in entries] * 2
        )
        current_globals = [int(entry["global_index"]) for entry in entries]
        current_frame_indices: list[int] = []
        current_patch_indices: list[int] = []
        current_group_sets: list[set[int]] = []
        current_descriptors: list[tuple[int, bool]] = []
        for entry_index, entry in enumerate(entries):
            for patch_index in entry["current_patches"]:
                for after in (False, True):
                    current_frame_indices.append(
                        entry_index + (batch if after else 0)
                    )
                    current_patch_indices.append(int(patch_index))
                    current_group_sets.append(set(entry["current_groups"]))
                    current_descriptors.append((entry_index, after))
        current_values = self.patch_bits_pairs(
            current_globals * 2,
            current_targets,
            current_previous,
            current_frame_indices,
            current_patch_indices,
            current_group_sets,
        )
        current_before = [0.0] * batch
        current_after = [0.0] * batch
        for value, (entry_index, after) in zip(
            current_values, current_descriptors
        ):
            (current_after if after else current_before)[entry_index] += value

        next_entries = [
            (entry_index, entry)
            for entry_index, entry in enumerate(entries)
            if int(entry["global_index"]) + 1 < N_TOTAL_PAIRS
        ]
        next_before = [0.0] * batch
        next_after = [0.0] * batch
        next_patch_counts = [0] * batch
        next_group_counts = [0] * batch
        if next_entries:
            next_batch = len(next_entries)
            next_globals = [
                int(entry["global_index"]) + 1 for _, entry in next_entries
            ]
            next_targets_base = [
                self._raw_frame(window_tokens, global_index, {})
                for global_index in next_globals
            ]
            before_previous = [entry["before"] for _, entry in next_entries]
            after_previous = [entry["candidate"] for _, entry in next_entries]
            context_previous = torch.stack(
                before_previous + after_previous
            ).to(device=self.device, dtype=torch.long)
            context_indices = torch.tensor(
                next_globals * 2, dtype=torch.long, device=self.device
            )
            before_after_context = self.model.prepare_frame_context(
                context_indices, context_previous
            )
            reshaped_context = []
            for value in before_after_context:
                if value is None:
                    reshaped_context.append(None)
                    continue
                expected = 2 * next_batch * self.sparse.patch_count
                if value.shape[0] != expected:
                    raise ValueError(
                        "unexpected batched HPAC context layout: "
                        f"{value.shape[0]} != {expected}"
                    )
                reshaped_context.append(value.reshape(
                    2 * next_batch,
                    self.sparse.patch_count,
                    *value.shape[1:],
                ))

            next_frame_indices: list[int] = []
            next_patch_indices: list[int] = []
            next_group_sets: list[set[int]] = []
            next_descriptors: list[tuple[int, bool]] = []
            for next_index, (entry_index, _) in enumerate(next_entries):
                affected = torch.zeros(
                    self.sparse.patch_count,
                    dtype=torch.bool,
                    device=self.device,
                )
                for value in reshaped_context:
                    if value is None:
                        continue
                    affected |= (
                        value[next_index] != value[next_index + next_batch]
                    ).reshape(self.sparse.patch_count, -1).any(dim=1)
                next_patches = affected.nonzero(as_tuple=False).flatten()
                context_positions: set[tuple[int, int]] = set()
                for value in reshaped_context:
                    if value is None or value.shape[-2:] == (1, 1):
                        continue
                    changed_context = (
                        value[next_index] != value[next_index + next_batch]
                    ).any(dim=1)
                    for patch_map in changed_context.index_select(
                        0, next_patches
                    ):
                        context_positions.update(map(
                            tuple,
                            patch_map.nonzero(as_tuple=False).tolist(),
                        ))
                next_groups = self._output_groups(context_positions)
                next_patch_counts[entry_index] = len(next_patches)
                next_group_counts[entry_index] = len(next_groups)
                for patch_index in next_patches.tolist():
                    for after in (False, True):
                        next_frame_indices.append(
                            next_index + (next_batch if after else 0)
                        )
                        next_patch_indices.append(int(patch_index))
                        next_group_sets.append(next_groups)
                        next_descriptors.append((entry_index, after))

            if next_frame_indices:
                next_targets = torch.stack(
                    next_targets_base + next_targets_base
                )
                next_values = self.patch_bits_pairs(
                    next_globals * 2,
                    next_targets,
                    torch.stack(before_previous + after_previous),
                    next_frame_indices,
                    next_patch_indices,
                    next_group_sets,
                    context=before_after_context,
                )
                for value, (entry_index, after) in zip(
                    next_values, next_descriptors
                ):
                    (next_after if after else next_before)[entry_index] += value

        for entry_index, entry in enumerate(entries):
            global_index = int(entry["global_index"])
            frame_deltas = {
                global_index: current_after[entry_index]
                - current_before[entry_index]
            }
            if global_index + 1 < N_TOTAL_PAIRS:
                frame_deltas[global_index + 1] = (
                    next_after[entry_index] - next_before[entry_index]
                )
            result_index = int(entry["result_index"])
            results[result_index] = (
                sum(frame_deltas.values()),
                frame_deltas,
                {
                    "current_patches": len(entry["current_patches"]),
                    "next_patches": next_patch_counts[entry_index],
                    "current_groups": len(entry["current_groups"]),
                    "next_groups": next_group_counts[entry_index],
                },
            )
        if any(result is None for result in results):
            raise RuntimeError("localized batch did not populate every result")
        return [result for result in results if result is not None]

    @torch.no_grad()
    def affected_frame_bits(
        self,
        window_tokens: torch.Tensor,
        local_indices: list[int],
        overrides: dict[int, torch.Tensor] | None = None,
        return_cost_tables: bool = False,
        cached_frame_bits: dict[int, float] | None = None,
    ) -> tuple[float, dict[int, torch.Tensor], dict[int, float]]:
        """Like affected_bits, plus reusable individual frame totals."""
        overrides = overrides or {}
        cached_frame_bits = cached_frame_bits or {}
        global_frames: set[int] = set()
        for local_index in local_indices:
            global_index = self.start_pair + local_index
            global_frames.add(global_index)
            if global_index + 1 < N_TOTAL_PAIRS:
                global_frames.add(global_index + 1)

        wanted_tables = set(local_indices) if return_cost_tables else set()
        tables: dict[int, torch.Tensor] = {}
        frame_totals: dict[int, float] = {}
        zero = torch.zeros_like(self.baseline_tokens[0])
        for global_index in sorted(global_frames):
            local_index = global_index - self.start_pair
            needs_table = local_index in wanted_tables
            if global_index in cached_frame_bits and not needs_table:
                frame_totals[global_index] = cached_frame_bits[global_index]
                continue
            target = self._raw_frame(window_tokens, global_index, overrides)
            previous = (
                zero
                if global_index == 0
                else self._raw_frame(window_tokens, global_index - 1, overrides)
            )
            bits, costs = self.frame_bits(
                global_index, target, previous, return_costs=needs_table
            )
            frame_totals[global_index] = bits
            if costs is not None:
                tables[local_index] = costs
        return sum(frame_totals.values()), tables, frame_totals

    @torch.no_grad()
    def affected_bits(
        self,
        window_tokens: torch.Tensor,
        local_indices: list[int],
        overrides: dict[int, torch.Tensor] | None = None,
        return_cost_tables: bool = False,
    ) -> tuple[float, dict[int, torch.Tensor]]:
        """Rate for changed frames plus their one-frame temporal dependents."""
        total, tables, _ = self.affected_frame_bits(
            window_tokens,
            local_indices,
            overrides=overrides,
            return_cost_tables=return_cost_tables,
        )
        return total, tables


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
    independent_alternatives: bool = False,
) -> tuple[list[TokenMove], dict[str, float | int | str]]:
    """Rank hard moves using perception gradients plus deployed-HPAC surprise."""
    if logits.grad is None and not rate_only:
        raise ValueError("logits must have a gradient before ranking moves")
    if max_pixels_per_frame < 1:
        raise ValueError("max_pixels_per_frame must be positive")
    if minimum_distance < 0:
        raise ValueError("minimum_distance cannot be negative")
    if logits.shape[:-1] != current_tokens.shape:
        raise ValueError("logits and token shapes do not match")
    if len(selected) != logits.shape[0]:
        raise ValueError("selected indices do not match the batch size")

    gradient = (
        torch.zeros_like(logits)
        if logits.grad is None
        else logits.grad.detach()
    )
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
    forbidden = F.one_hot(current, N_TOKENS).bool()
    if attempted_masks is not None:
        attempted = torch.stack(
            [attempted_masks[index] for index in selected]
        ).to(logits.device)
        if attempted.dtype == torch.bool:
            forbidden |= attempted.unsqueeze(-1)
        else:
            category_bits = 1 << torch.arange(N_TOKENS, device=logits.device)
            forbidden |= attempted.unsqueeze(-1).bitwise_and(category_bits).bool()
    alternative_objective, alternative_token = objective.masked_fill(
        forbidden, torch.inf
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
        if independent_alternatives:
            category_benefit = (
                current_objective[batch_index].unsqueeze(-1)
                - objective[batch_index]
            ).masked_fill(forbidden[batch_index], -torch.inf)
            flat_benefit = category_benefit.reshape(-1)
            positive = flat_benefit > 0
            positive_count = int(positive.sum())
            positive_candidates += positive_count
            if not positive_count:
                continue
            keep_count = min(positive_count, max_pixels_per_frame)
            ranked = flat_benefit.masked_fill(
                ~positive, -torch.inf
            ).topk(keep_count, sorted=True).indices
            width = current_tokens.shape[2]
            for flat_index_tensor in ranked:
                flat_index = int(flat_index_tensor)
                pixel_index, after = divmod(flat_index, N_TOKENS)
                row, col = divmod(pixel_index, width)
                before = int(current_tokens[batch_index, row, col])
                moves.append(TokenMove(
                    batch_index=batch_index,
                    parameter_index=parameter_index,
                    row=row,
                    col=col,
                    before=before,
                    after=after,
                    benefit=float(category_benefit[row, col, after]),
                    perception_benefit=float(
                        gradient[batch_index, row, col, before]
                        - gradient[batch_index, row, col, after]
                    ),
                    direct_rate_benefit_bits=float(
                        rate_bits[batch_index, row, col, before]
                        - rate_bits[batch_index, row, col, after]
                    ),
                ))
                if attempted_masks is not None:
                    if attempted_masks[parameter_index].dtype == torch.bool:
                        attempted_masks[parameter_index][row, col] = True
                    else:
                        attempted_masks[parameter_index][row, col] |= 1 << after
            continue

        flat_benefit = benefit[batch_index].reshape(-1)
        positive = flat_benefit > 0
        if attempted_masks is not None and attempted_masks[parameter_index].dtype == torch.bool:
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
                if attempted_masks[parameter_index].dtype == torch.bool:
                    attempted_masks[parameter_index][row, col] = True
                else:
                    attempted_masks[parameter_index][row, col] |= 1 << after
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
        "independent_alternatives": independent_alternatives,
    }


def rank_token_regions(
    current_tokens: torch.Tensor,
    selected: list[int],
    max_regions_per_frame: int,
    attempted_masks: list[torch.Tensor],
    category_rate_bits: torch.Tensor,
    shapes: tuple[tuple[int, int], ...],
    minimum_distance: int = 0,
) -> tuple[list[TokenRegionMove], dict[str, float | int | str]]:
    """Rank small constant-category regions by summed direct HPAC benefit."""
    if max_regions_per_frame < 1 or not shapes:
        raise ValueError("region count and shapes must be non-empty")
    if category_rate_bits.shape[:-1] != current_tokens.shape:
        raise ValueError("category rate table does not match tokens")
    if len(selected) != len(current_tokens):
        raise ValueError("selected indices do not match the batch size")

    current = current_tokens.to(category_rate_bits.device).long()
    current_rate = category_rate_bits.gather(
        -1, current.unsqueeze(-1)
    ).squeeze(-1)
    pixel_benefit = (
        current_rate.unsqueeze(-1) - category_rate_bits
    ).permute(0, 3, 1, 2)
    same_category = F.one_hot(current, N_TOKENS).permute(0, 3, 1, 2).float()
    batch, _, height, width = pixel_benefit.shape
    best = torch.full_like(pixel_benefit, -torch.inf)
    best_shape = torch.full(
        (batch, N_TOKENS, height, width),
        -1,
        dtype=torch.int16,
        device=pixel_benefit.device,
    )
    for shape_index, (region_height, region_width) in enumerate(shapes):
        if region_height < 1 or region_width < 1:
            raise ValueError("region dimensions must be positive")
        if region_height > height or region_width > width:
            continue
        area = region_height * region_width
        score = F.avg_pool2d(
            pixel_benefit,
            (region_height, region_width),
            stride=1,
        ) * area
        unchanged = F.avg_pool2d(
            same_category,
            (region_height, region_width),
            stride=1,
        ) * area
        score = score.masked_fill(unchanged == area, -torch.inf)
        valid_height, valid_width = score.shape[-2:]
        destination = best[:, :, :valid_height, :valid_width]
        better = score > destination
        destination.copy_(torch.where(better, score, destination))
        shape_destination = best_shape[:, :, :valid_height, :valid_width]
        shape_destination.copy_(torch.where(
            better,
            torch.full_like(shape_destination, shape_index),
            shape_destination,
        ))

    attempted = torch.stack(
        [attempted_masks[index] for index in selected]
    ).to(best.device)
    category_bits = 1 << torch.arange(N_TOKENS, device=best.device)
    best.masked_fill_(
        attempted.unsqueeze(1).bitwise_and(
            category_bits.view(1, -1, 1, 1)
        ).bool(),
        -torch.inf,
    )

    moves: list[TokenRegionMove] = []
    positive_candidates = 0
    for batch_index, parameter_index in enumerate(selected):
        flat_scores = best[batch_index].reshape(-1)
        positive = flat_scores > 0
        positive_count = int(positive.sum())
        positive_candidates += positive_count
        if not positive_count:
            continue
        pool_size = min(positive_count, max_regions_per_frame * 64)
        ranked = flat_scores.masked_fill(
            ~positive, -torch.inf
        ).topk(pool_size, sorted=True).indices
        chosen_anchors: list[tuple[int, int]] = []
        for flat_index_tensor in ranked:
            flat_index = int(flat_index_tensor)
            after, anchor_index = divmod(flat_index, height * width)
            row, col = divmod(anchor_index, width)
            if minimum_distance and any(
                max(abs(row - other_row), abs(col - other_col))
                < minimum_distance
                for other_row, other_col in chosen_anchors
            ):
                continue
            shape_index = int(best_shape[batch_index, after, row, col])
            if shape_index < 0:
                continue
            region_height, region_width = shapes[shape_index]
            moves.append(TokenRegionMove(
                batch_index=batch_index,
                parameter_index=parameter_index,
                row=row,
                col=col,
                height=region_height,
                width=region_width,
                after=after,
                benefit=float(best[batch_index, after, row, col]),
                direct_rate_benefit_bits=float(
                    best[batch_index, after, row, col]
                ),
            ))
            chosen_anchors.append((row, col))
            attempted_masks[parameter_index][row, col] |= 1 << after
            if len(chosen_anchors) == max_regions_per_frame:
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
        "proposal_mode": "rate-regions",
        "independent_alternatives": True,
        "region_shapes": ",".join(f"{h}x{w}" for h, w in shapes),
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
