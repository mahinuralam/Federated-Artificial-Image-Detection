"""Federated averaging utilities — weight scaling, aggregation, and monitoring."""
from __future__ import annotations

import numpy as np


def scale_weights(
    weights: list[np.ndarray], scalar: float
) -> list[np.ndarray]:
    """Multiply every weight tensor by a scalar."""
    return [w * scalar for w in weights]


def clip_weights(
    weights: list[np.ndarray], percentile: int = 95
) -> list[np.ndarray]:
    """
    Per-layer percentile clipping.

    Clips each weight layer to ±p-th percentile of its absolute values,
    preventing large outlier updates from destabilising aggregation.
    """
    clipped = []
    for layer in weights:
        threshold = np.percentile(np.abs(layer), percentile)
        clipped.append(np.clip(layer, -threshold, threshold))
    return clipped


def federated_average(
    scaled_weight_list: list[list[np.ndarray]],
    threshold_percentile: int = 95,
) -> list[np.ndarray]:
    """
    FedAvg aggregation: sum pre-scaled client weights, then clip extremes.

    Callers are responsible for scaling each client's weights by N_k / N
    before passing them here.
    """
    averaged = [np.zeros_like(w) for w in scaled_weight_list[0]]
    for client_weights in scaled_weight_list:
        for i, layer in enumerate(client_weights):
            averaged[i] += layer
    return clip_weights(averaged, threshold_percentile)


def weight_divergence(
    old_weights: list[np.ndarray], new_weights: list[np.ndarray]
) -> float:
    """L2 norm of the element-wise difference between two weight lists."""
    return float(
        np.sqrt(sum(np.sum((o - n) ** 2) for o, n in zip(old_weights, new_weights)))
    )
