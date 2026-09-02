"""Learned per-output-channel bit depths for the portable integer HPAC."""

from __future__ import annotations

import math
import types
from collections import Counter

import torch
import torch.nn.functional as F

from hpac_integer import IntegerConv2d, IntegerLinear, ste_round


COMPRESSIBLE_TYPES = (IntegerConv2d, IntegerLinear)


def _channel_view(module, value: torch.Tensor) -> torch.Tensor:
    return value.view(value.shape[0], *([1] * (module.weight.ndim - 1)))


def _quantized_weight(module) -> torch.Tensor:
    bits = F.relu(module.bit_depth)
    bits = bits.clamp(max=module.self_compress_max_bits)
    if module.self_compress_deployed:
        bits = ste_round(bits)
    radius = torch.pow(2.0, bits - 1.0)
    low = _channel_view(
        module,
        torch.maximum(-radius, radius.new_full(radius.shape, -module.weight_bound)),
    )
    high = _channel_view(
        module,
        torch.minimum(
            radius - 1.0, radius.new_full(radius.shape, module.weight_bound)
        ),
    )
    weight = ste_round(torch.maximum(torch.minimum(module.weight, high), low))
    if isinstance(module, IntegerConv2d):
        weight = weight * module.mask
    return weight


def _codes(module):
    weight = _quantized_weight(module)
    bias = ste_round(module.bias.clamp(-32768, 32767))
    exponent = None
    if hasattr(module, "exponent"):
        exponent = ste_round(module.exponent.clamp(module.exponent_min, 0))
    return weight, bias, exponent


def enable_self_compression(model, init_bits: float = 8.0) -> None:
    """Attach differentiable channel bit depths without changing model topology."""
    for module in model.modules():
        if not isinstance(module, COMPRESSIBLE_TYPES):
            continue
        max_bits = int(math.ceil(math.log2(2 * module.weight_bound + 1)))
        module.self_compress_max_bits = max_bits
        module.self_compress_deployed = False
        if not hasattr(module, "bit_depth"):
            module.register_parameter(
                "bit_depth",
                torch.nn.Parameter(module.weight.new_full(
                    (module.weight.shape[0],), init_bits
                )),
            )
        module.codes = types.MethodType(_codes, module)


def set_deployed_bit_depths(model, deployed: bool) -> None:
    for module in model.modules():
        if isinstance(module, COMPRESSIBLE_TYPES) and hasattr(module, "bit_depth"):
            module.self_compress_deployed = deployed


def _weights_per_output(module) -> torch.Tensor:
    if isinstance(module, IntegerConv2d):
        mask = module.mask.to(torch.bool).expand_as(module.weight)
        return mask.reshape(mask.shape[0], -1).sum(dim=1).to(module.weight.dtype)
    return torch.full(
        (module.weight.shape[0],),
        module.weight[0].numel(),
        dtype=module.weight.dtype,
        device=module.weight.device,
    )


def variable_weight_bits(model, deployed: bool = False) -> torch.Tensor:
    total = None
    for module in model.modules():
        if not isinstance(module, COMPRESSIBLE_TYPES) or not hasattr(module, "bit_depth"):
            continue
        bits = F.relu(module.bit_depth)
        if deployed:
            bits = bits.clamp(max=module.self_compress_max_bits).round()
        layer_bits = (bits * _weights_per_output(module)).sum()
        total = layer_bits if total is None else total + layer_bits
    if total is None:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    return total


def fixed_and_metadata_bits(model) -> int:
    """Bits outside compressible weights, including 4-bit channel descriptors."""
    modules = dict(model.named_modules())
    total = 0
    channels = 0
    for name, parameter in model.named_parameters():
        module_name, field = name.rsplit(".", 1)
        module = modules[module_name]
        if field == "bit_depth":
            channels += parameter.numel()
            continue
        if field == "weight" and isinstance(module, COMPRESSIBLE_TYPES):
            continue
        total += parameter.numel() * (16 if field == "bias" else 8)
    return total + channels * 4


def bit_depth_histogram(model) -> dict[str, int]:
    counts: Counter[int] = Counter()
    for module in model.modules():
        if not isinstance(module, COMPRESSIBLE_TYPES) or not hasattr(module, "bit_depth"):
            continue
        values = (
            F.relu(module.bit_depth.detach())
            .clamp(max=module.self_compress_max_bits)
            .round()
            .to(torch.int64)
            .cpu()
            .tolist()
        )
        counts.update(values)
    return {str(bits): counts[bits] for bits in sorted(counts)}


def estimated_model_bits(model) -> int:
    return fixed_and_metadata_bits(model) + math.ceil(
        float(variable_weight_bits(model, deployed=True).detach().cpu())
    )
