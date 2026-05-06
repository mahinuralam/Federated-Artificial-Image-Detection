"""Tests for federated averaging utilities."""
import numpy as np
import pytest

from fmacnn.federated import (
    clip_weights,
    federated_average,
    scale_weights,
    weight_divergence,
)


def _rand_weights(shapes=((10, 10), (10,), (5, 10), (5,))):
    return [np.random.randn(*s).astype("float32") for s in shapes]


def test_scale_weights():
    w = _rand_weights()
    scaled = scale_weights(w, 0.5)
    for orig, sc in zip(w, scaled):
        np.testing.assert_allclose(sc, orig * 0.5)


def test_federated_average_uniform():
    """With equal scaling (1/N each), FedAvg should return the element-wise mean."""
    n = 4
    weight_lists = [_rand_weights() for _ in range(n)]
    scaled = [scale_weights(wl, 1.0 / n) for wl in weight_lists]

    # Use 100th percentile so clipping doesn't alter the result
    avg = federated_average(scaled, threshold_percentile=100)

    for i, layer_avg in enumerate(avg):
        expected = np.mean([wl[i] for wl in weight_lists], axis=0)
        np.testing.assert_allclose(layer_avg, expected, atol=1e-5)


def test_clip_weights_bounds():
    w = [np.array([-10.0, -1.0, 0.0, 1.0, 10.0])]
    clipped = clip_weights(w, percentile=80)
    # All values should be within the clipped range
    threshold = np.percentile(np.abs(w[0]), 80)
    assert np.all(np.abs(clipped[0]) <= threshold + 1e-6)


def test_weight_divergence_identical():
    w = _rand_weights()
    assert weight_divergence(w, w) == pytest.approx(0.0, abs=1e-6)


def test_weight_divergence_known():
    w1 = [np.array([1.0, 0.0])]
    w2 = [np.array([0.0, 0.0])]
    # L2 norm of [1, 0] = 1.0
    assert weight_divergence(w1, w2) == pytest.approx(1.0, abs=1e-6)
