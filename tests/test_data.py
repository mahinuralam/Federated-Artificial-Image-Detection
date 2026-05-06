"""Tests for data utilities."""
import numpy as np
import pytest

from fmacnn.data import batch_data, create_clients, split_dataset


def _dummy_dataset(n=200, h=32, w=32, c=3, num_classes=2):
    images = np.random.rand(n, h, w, c).astype("float32")
    labels = np.eye(num_classes)[np.random.randint(0, num_classes, n)]
    return images, labels


def test_split_sizes():
    images, labels = _dummy_dataset(200)
    X_train, X_test, y_train, y_test = split_dataset(images, labels, test_size=0.1)
    assert len(X_train) == 180
    assert len(X_test) == 20


def test_create_clients_iid_count():
    images, labels = _dummy_dataset(100)
    clients = create_clients(images, labels, num_clients=10, iid=True)
    assert len(clients) == 10
    assert all(len(v) == 10 for v in clients.values())


def test_create_clients_noniid_sorted():
    images, labels = _dummy_dataset(100, num_classes=2)
    clients = create_clients(images, labels, num_clients=10, iid=False)
    # First clients should be biased toward class 0
    first_client = clients["client_1"]
    class_indices = [np.argmax(y) for _, y in first_client]
    # At least majority should be class 0 in a sorted non-IID split
    assert class_indices.count(0) >= class_indices.count(1)


def test_batch_data_shapes():
    images, labels = _dummy_dataset(50)
    shard = list(zip(images, labels))
    ds = batch_data(shard, batch_size=16)
    batches = list(ds)
    assert len(batches) > 0
    x_batch, y_batch = batches[0]
    assert x_batch.shape[1:] == (32, 32, 3)


def test_client_sizes_sum_to_train():
    images, labels = _dummy_dataset(100)
    clients = create_clients(images, labels, num_clients=10, iid=True)
    total = sum(len(v) for v in clients.values())
    assert total == 100
