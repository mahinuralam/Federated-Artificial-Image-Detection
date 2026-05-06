"""Dataset loading and federated client data partitioning."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

_VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def load_dataset(
    data_root: str | Path,
    target_size: tuple[int, int] = (256, 256),
    color_mode: str = "rgb",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load images from a class-folder layout, normalise to [0, 1], one-hot
    encode labels.

    Expected layout:
        data_root/
            class_a/image1.jpg
            class_b/image2.png
            ...

    Returns:
        images:  float32 array of shape (N, H, W, C)
        labels:  one-hot encoded int array of shape (N, num_classes)
        classes: sorted class-name array from LabelBinarizer
    """
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {data_root}")

    images, labels = [], []

    for class_dir in sorted(data_root.iterdir()):
        if not class_dir.is_dir():
            continue
        for img_path in class_dir.glob("*"):
            if img_path.suffix.lower() not in _VALID_SUFFIXES:
                continue
            img = tf.keras.utils.load_img(
                img_path, target_size=target_size, color_mode=color_mode
            )
            images.append(tf.keras.utils.img_to_array(img) / 255.0)
            labels.append(class_dir.name)

    if not images:
        raise ValueError(f"No images found under {data_root}")

    image_array = np.stack(images).astype("float32")
    lb = LabelBinarizer()
    label_array = lb.fit_transform(labels)
    if label_array.ndim == 1:           # binary case → add class axis
        label_array = label_array[:, np.newaxis]

    return image_array, label_array, lb.classes_


def split_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.1,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified 90/10 train/test split."""
    return train_test_split(
        images, labels, test_size=test_size, random_state=random_state
    )


def create_clients(
    images: np.ndarray,
    labels: np.ndarray,
    num_clients: int = 100,
    iid: bool = True,
    prefix: str = "client",
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """
    Shard (image, label) pairs into per-client lists.

    IID:     data is randomly shuffled before sharding → uniform class mix.
    Non-IID: data is sorted by class label before sharding → skewed class
             distribution that mimics real-world heterogeneous clients.
    """
    paired: list[tuple[np.ndarray, np.ndarray]] = list(zip(images, labels))

    if iid:
        random.shuffle(paired)
    else:
        paired.sort(key=lambda xy: int(np.argmax(xy[1])))

    size = len(paired) // num_clients
    shards = [paired[i : i + size] for i in range(0, size * num_clients, size)]
    names = [f"{prefix}_{i + 1}" for i in range(num_clients)]
    assert len(shards) == len(names), "Mismatch between shard count and client count"

    return dict(zip(names, shards))


def batch_data(
    shard: list[tuple[np.ndarray, np.ndarray]],
    batch_size: int = 32,
) -> tf.data.Dataset:
    """Convert a client shard into a shuffled, prefetched TF Dataset."""
    xs = np.stack([x for x, _ in shard]).astype("float32")
    ys = np.stack([y for _, y in shard]).astype("float32")
    return (
        tf.data.Dataset.from_tensor_slices((xs, ys))
        .shuffle(buffer_size=len(shard))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def build_test_dataset(
    X_test: np.ndarray, y_test: np.ndarray
) -> tf.data.Dataset:
    """Wrap test arrays into a single-batch TF Dataset."""
    return tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(len(y_test))
