import math

import torch


def _value(outputs):
    cumulative = outputs.sigmoid()
    return cumulative.sum(dim=-1)


def _subset_size(subset):
    return subset.bit_count()


def shapley_values(values, modality_count):
    sample_count = values.size(0)
    result = values.new_zeros(sample_count, modality_count)
    factorials = [math.factorial(index) for index in range(modality_count + 1)]
    denominator = math.factorial(modality_count)
    full_mask = (1 << modality_count) - 1
    for index in range(modality_count):
        without = full_mask ^ (1 << index)
        for subset in range(1 << modality_count):
            if subset & (1 << index):
                continue
            size = _subset_size(subset)
            weight = factorials[size] * factorials[modality_count - size - 1] / denominator
            result[:, index] += weight * (values[:, subset | (1 << index)] - values[:, subset])
    return result


def interaction_values(values, modality_count):
    sample_count = values.size(0)
    result = values.new_zeros(sample_count, modality_count, modality_count)
    factorials = [math.factorial(index) for index in range(modality_count + 1)]
    denominator = 2 * math.factorial(modality_count - 1)
    full_mask = (1 << modality_count) - 1
    for first in range(modality_count):
        for second in range(first + 1, modality_count):
            available = full_mask ^ (1 << first) ^ (1 << second)
            for subset in range(1 << modality_count):
                if subset & ~available:
                    continue
                size = _subset_size(subset)
                weight = factorials[size] * factorials[modality_count - size - 2] / denominator
                difference = (
                    values[:, subset | (1 << first) | (1 << second)]
                    - values[:, subset | (1 << first)]
                    - values[:, subset | (1 << second)]
                    + values[:, subset]
                )
                result[:, first, second] += weight * difference
                result[:, second, first] = result[:, first, second]
    return result


def intershap_summary(shapley, off_diagonal, tolerance=1e-8):
    """Full SII and InterSHAP equations 4-11, adapted to one scalar coalition game."""
    full_sii = off_diagonal.clone()
    full_sii.diagonal(dim1=-2, dim2=-1).copy_(shapley - off_diagonal.sum(dim=-1))
    off_mask = ~torch.eye(shapley.size(1), dtype=torch.bool, device=shapley.device)
    global_phi = full_sii.mean(dim=0).abs()
    global_denominator = global_phi.sum().item()
    global_ratio = global_phi[off_mask].sum().item() / global_denominator if global_denominator > tolerance else None
    local_absolute = full_sii.abs()
    local_denominators = local_absolute.sum(dim=(1, 2))
    local_numerators = local_absolute[:, off_mask].sum(dim=1)
    local_ratios = [numerator.item() / denominator.item() if denominator.item() > tolerance else None
                    for numerator, denominator in zip(local_numerators, local_denominators)]
    defined = [ratio for ratio in local_ratios if ratio is not None]
    return full_sii, {
        "global_abs_mean_sii": global_phi.tolist(),
        "global_intershap": global_ratio,
        "global_undefined_count": int(global_ratio is None),
        "local_intershap": local_ratios,
        "local_mean_intershap_defined": sum(defined) / len(defined) if defined else None,
        "local_defined_count": len(defined),
        "local_undefined_count": len(local_ratios) - len(defined),
        "denominator_tolerance": tolerance,
        "global_formula": "sum_off(abs(mean_samples(full_sii))) / sum_all(abs(mean_samples(full_sii)))",
        "local_formula": "mean_defined_samples(sum_off(abs(full_sii_sample)) / sum_all(abs(full_sii_sample)))",
    }


def explain_coalitions(logits):
    values = _value(logits)
    modality_count = values.size(1).bit_length() - 1
    shapley = shapley_values(values, modality_count)
    interactions = interaction_values(values, modality_count)
    return values, shapley, interactions
