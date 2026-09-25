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


def explain_coalitions(logits):
    values = _value(logits)
    modality_count = values.size(1).bit_length() - 1
    shapley = shapley_values(values, modality_count)
    interactions = interaction_values(values, modality_count)
    return values, shapley, interactions
