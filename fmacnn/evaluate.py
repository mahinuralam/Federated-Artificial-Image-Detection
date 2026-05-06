"""Model evaluation utilities."""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score


def evaluate(
    model: tf.keras.Model,
    X_test: np.ndarray,
    Y_test: np.ndarray,
) -> tuple[float, float]:
    """
    Run inference and return (accuracy, categorical_crossentropy_loss).

    Args:
        model:  compiled or bare Keras model
        X_test: float32 image array (N, H, W, C)
        Y_test: one-hot label array  (N, num_classes)

    Returns:
        (accuracy, loss) as plain Python floats
    """
    preds = model.predict(X_test, verbose=0)
    cce = tf.keras.losses.CategoricalCrossentropy()
    loss = float(cce(Y_test, preds).numpy())
    acc = float(accuracy_score(np.argmax(Y_test, axis=1), np.argmax(preds, axis=1)))
    return acc, loss
