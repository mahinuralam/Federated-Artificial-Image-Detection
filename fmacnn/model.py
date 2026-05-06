"""MACNN — Multi-Attention CNN architecture."""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


# ---------------------------------------------------------------------------
# Attention blocks
# ---------------------------------------------------------------------------

def _se_block(x: tf.Tensor) -> tf.Tensor:
    """Squeeze-and-Excitation channel attention."""
    filters = x.shape[-1]
    reduction = max(filters // 16, 1)
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(reduction, activation="relu", kernel_initializer="he_normal")(se)
    se = layers.Dense(filters, activation="sigmoid", kernel_initializer="he_normal")(se)
    se = layers.Reshape((1, 1, filters))(se)
    return layers.Multiply()([x, se])


def _spatial_attention(x: tf.Tensor) -> tf.Tensor:
    """CBAM-style spatial attention (avg + max pooling along channel axis)."""
    avg = layers.Lambda(lambda t: tf.reduce_mean(t, axis=3, keepdims=True))(x)
    max_ = layers.Lambda(lambda t: tf.reduce_max(t, axis=3, keepdims=True))(x)
    concat = layers.Concatenate(axis=3)([avg, max_])
    attn = layers.Conv2D(
        1, kernel_size=7, padding="same", activation="sigmoid",
        kernel_initializer="he_normal",
    )(concat)
    return layers.Multiply()([x, attn])


def _channel_attention(x: tf.Tensor, ratio: int = 8) -> tf.Tensor:
    """Dual-pooling channel attention with shared dense weights."""
    ch = x.shape[-1]
    reduction = max(ch // ratio, 1)
    dense1 = layers.Dense(reduction, activation="relu", kernel_initializer="he_normal")
    dense2 = layers.Dense(ch, activation="sigmoid", kernel_initializer="he_normal")

    avg = dense2(dense1(layers.GlobalAveragePooling2D()(x)))
    max_ = dense2(dense1(layers.GlobalMaxPooling2D()(x)))
    attn = layers.Reshape((1, 1, ch))(layers.Add()([avg, max_]))
    return layers.Multiply()([x, attn])


# ---------------------------------------------------------------------------
# Convolutional block helper
# ---------------------------------------------------------------------------

def _double_conv(x: tf.Tensor, filters: int, reg: regularizers.Regularizer) -> tf.Tensor:
    for _ in range(2):
        x = layers.Conv2D(
            filters, 3, padding="same", activation="relu",
            kernel_initializer="he_normal", kernel_regularizer=reg,
        )(x)
        x = layers.BatchNormalization()(x)
    return x


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def build_macnn(
    input_shape: tuple,
    num_classes: int,
    l2_reg: float = 1e-4,
) -> models.Model:
    """
    Build the Multi-Attention CNN.

    Architecture:
      Block 1 (32 filters):  double-conv → pool → dropout → SE → spatial-attn
      Block 2 (64 filters):  double-conv → pool → dropout → SE → channel-attn
      Block 3 (128 filters): double-conv → pool → dropout
      Head:                  flatten → Dense(512) → Dense(256) → softmax
    """
    reg = regularizers.l2(l2_reg)
    inputs = layers.Input(shape=input_shape)

    # Block 1
    x = _double_conv(inputs, 32, reg)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)
    x = _se_block(x)
    x = _spatial_attention(x)

    # Block 2
    x = _double_conv(x, 64, reg)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.30)(x)
    x = _se_block(x)
    x = _channel_attention(x)

    # Block 3 (no attention — deepest features need no further gating)
    x = _double_conv(x, 128, reg)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.40)(x)

    # Classification head
    x = layers.Flatten()(x)
    for units in [512, 256]:
        x = layers.Dense(
            units, activation="relu",
            kernel_initializer="he_normal", kernel_regularizer=reg,
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(
        num_classes, activation="softmax",
        kernel_initializer="glorot_uniform",
    )(x)

    return models.Model(inputs, outputs, name="MACNN")
