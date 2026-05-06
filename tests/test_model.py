"""Tests for model architecture."""
import numpy as np
import pytest
import tensorflow as tf

from fmacnn.model import build_macnn


def test_output_shape_binary():
    model = build_macnn(input_shape=(64, 64, 3), num_classes=2)
    x = np.random.rand(4, 64, 64, 3).astype("float32")
    out = model.predict(x, verbose=0)
    assert out.shape == (4, 2)


def test_output_shape_multiclass():
    model = build_macnn(input_shape=(32, 32, 3), num_classes=5)
    x = np.random.rand(2, 32, 32, 3).astype("float32")
    out = model.predict(x, verbose=0)
    assert out.shape == (2, 5)


def test_softmax_sums_to_one():
    model = build_macnn(input_shape=(32, 32, 3), num_classes=3)
    x = np.random.rand(8, 32, 32, 3).astype("float32")
    out = model.predict(x, verbose=0)
    np.testing.assert_allclose(out.sum(axis=1), np.ones(8), atol=1e-5)


def test_get_set_weights_roundtrip():
    model = build_macnn(input_shape=(32, 32, 3), num_classes=2)
    original = model.get_weights()
    perturbed = [w + 0.01 for w in original]
    model.set_weights(perturbed)
    retrieved = model.get_weights()
    for orig, ret in zip(perturbed, retrieved):
        np.testing.assert_allclose(orig, ret, atol=1e-6)
