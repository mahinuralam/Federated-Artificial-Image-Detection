"""Federated training loops for IID and Non-IID settings."""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import SGD

from .config import Config
from .evaluate import evaluate
from .federated import clip_weights, federated_average, scale_weights, weight_divergence
from .model import build_macnn

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class RoundResult:
    accuracy: float
    loss: float
    divergence: float
    elapsed_seconds: float


@dataclass
class TrainResult:
    rounds: list[RoundResult] = field(default_factory=list)

    @property
    def accuracies(self) -> list[float]:
        return [r.accuracy for r in self.rounds]

    @property
    def losses(self) -> list[float]:
        return [r.loss for r in self.rounds]

    @property
    def divergences(self) -> list[float]:
        return [r.divergence for r in self.rounds]

    @property
    def per_round_times(self) -> list[float]:
        return [r.elapsed_seconds for r in self.rounds]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _current_lr(initial_lr: float, comm_round: int, decay_rate: float, decay_steps: int) -> float:
    return initial_lr * (decay_rate ** (comm_round // decay_steps))


def _local_train(
    global_weights: list[np.ndarray],
    dataset: tf.data.Dataset,
    input_shape: tuple,
    num_classes: int,
    l2_reg: float,
    local_epochs: int,
    optimizer: SGD,
) -> list[np.ndarray]:
    """Train a fresh local model from global weights and return its weights."""
    local_model = build_macnn(input_shape, num_classes, l2_reg)
    local_model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    local_model.set_weights(global_weights)
    local_model.fit(dataset, epochs=local_epochs, verbose=0)
    weights = local_model.get_weights()
    del local_model
    tf.keras.backend.clear_session()
    return weights


def _select_clients(
    all_clients: list[str], k: int
) -> list[str]:
    return random.sample(all_clients, k=min(k, len(all_clients)))


def _eval_on_test(
    model: tf.keras.Model, test_batched: tf.data.Dataset
) -> tuple[float, float]:
    for X_batch, Y_batch in test_batched:
        return evaluate(model, X_batch.numpy(), Y_batch.numpy())
    raise RuntimeError("test_batched dataset is empty")


# ---------------------------------------------------------------------------
# IID training
# ---------------------------------------------------------------------------

def train_iid(
    global_model: tf.keras.Model,
    clients_batched: dict[str, tf.data.Dataset],
    client_sizes: dict[str, int],
    test_batched: tf.data.Dataset,
    config: Config,
    input_shape: tuple,
    num_classes: int,
) -> TrainResult:
    """
    Federated training under IID data distribution.

    client_sizes: mapping of client name → number of samples (used for
                  proper N_k / N_selected scaling within each round).
    """
    cfg = config.iid
    fed = config.federated
    result = TrainResult()
    all_clients = list(clients_batched.keys())

    log.info("=" * 60)
    log.info("FEDERATED LEARNING — IID")
    log.info(
        "clients=%d  per_round=%d  rounds=%d  local_epochs=%d  lr=%.4f",
        len(all_clients), cfg.num_clients_per_round, cfg.comms_round,
        cfg.local_epochs, cfg.initial_lr,
    )
    log.info("=" * 60)

    for comm_round in range(cfg.comms_round):
        t0 = time.time()
        global_weights = global_model.get_weights()
        selected = _select_clients(all_clients, cfg.num_clients_per_round)

        # Scale by share of selected-client data (factors sum to 1)
        selected_total = sum(client_sizes[c] for c in selected)
        scaled_updates: list[list[np.ndarray]] = []

        for client in selected:
            lr = _current_lr(cfg.initial_lr, comm_round, cfg.lr_decay_rate, cfg.lr_decay_steps)
            opt = SGD(learning_rate=lr, momentum=fed.momentum, nesterov=fed.nesterov)

            local_weights = _local_train(
                global_weights,
                clients_batched[client],
                input_shape,
                num_classes,
                config.model.l2_reg,
                cfg.local_epochs,
                opt,
            )
            scale = client_sizes[client] / selected_total
            scaled_updates.append(scale_weights(local_weights, scale))

            if config.debug:
                log.debug("round=%d  client=%s  scale=%.4f", comm_round, client, scale)

        new_weights = federated_average(scaled_updates, fed.threshold_percentile)
        div = weight_divergence(global_weights, new_weights)
        global_model.set_weights(new_weights)

        acc, loss = _eval_on_test(global_model, test_batched)
        elapsed = time.time() - t0
        result.rounds.append(RoundResult(acc, loss, div, elapsed))

        if (comm_round + 1) % 10 == 0:
            log.info(
                "[IID %d/%d]  acc=%.4f  loss=%.4f  div=%.4f  time=%.1fs",
                comm_round + 1, cfg.comms_round, acc, loss, div, elapsed,
            )

    log.info("IID complete — final acc=%.4f  loss=%.4f", result.accuracies[-1], result.losses[-1])
    return result


# ---------------------------------------------------------------------------
# Non-IID training
# ---------------------------------------------------------------------------

def train_noniid(
    global_model: tf.keras.Model,
    clients_batched: dict[str, tf.data.Dataset],
    client_sizes: dict[str, int],
    test_batched: tf.data.Dataset,
    config: Config,
    input_shape: tuple,
    num_classes: int,
) -> TrainResult:
    """
    Federated training under Non-IID data distribution.

    Supports optional early-stopping via target_accuracy, max_total_minutes,
    and per-client batch capping (max_batches_per_client) for fast iteration.
    """
    cfg = config.noniid
    fed = config.federated
    result = TrainResult()
    all_clients = list(clients_batched.keys())
    overall_start = time.time()

    log.info("=" * 60)
    log.info("FEDERATED LEARNING — NON-IID")
    log.info(
        "clients=%d  per_round=%d  rounds=%d  local_epochs=%d  lr=%.4f",
        len(all_clients), cfg.num_clients_per_round, cfg.comms_round,
        cfg.local_epochs, cfg.initial_lr,
    )
    if cfg.max_total_minutes:
        log.info("time budget: %.1f min", cfg.max_total_minutes)
    if cfg.target_accuracy:
        log.info("early-stop accuracy: %.4f", cfg.target_accuracy)
    log.info("=" * 60)

    stop_reason: str | None = None

    try:
        for comm_round in range(cfg.comms_round):
            t0 = time.time()
            global_weights = global_model.get_weights()
            selected = _select_clients(all_clients, cfg.num_clients_per_round)

            # Effective sizes respect the optional batch cap
            if cfg.max_batches_per_client is not None:
                eff_sizes = {
                    c: min(client_sizes[c], cfg.max_batches_per_client * cfg.batch_size)
                    for c in selected
                }
            else:
                eff_sizes = {c: client_sizes[c] for c in selected}

            selected_total = max(sum(eff_sizes.values()), 1)
            scaled_updates: list[list[np.ndarray]] = []

            for client in selected:
                lr = _current_lr(cfg.initial_lr, comm_round, cfg.lr_decay_rate, cfg.lr_decay_steps)
                opt = SGD(learning_rate=lr, momentum=fed.momentum, nesterov=fed.nesterov)

                dataset = clients_batched[client]
                if cfg.max_batches_per_client is not None:
                    dataset = dataset.take(cfg.max_batches_per_client)

                local_weights = _local_train(
                    global_weights,
                    dataset,
                    input_shape,
                    num_classes,
                    config.model.l2_reg,
                    cfg.local_epochs,
                    opt,
                )
                scale = eff_sizes[client] / selected_total
                scaled_updates.append(scale_weights(local_weights, scale))

                if config.debug:
                    log.debug("round=%d  client=%s  scale=%.4f", comm_round, client, scale)

            new_weights = federated_average(scaled_updates, fed.threshold_percentile)
            div = weight_divergence(global_weights, new_weights)
            global_model.set_weights(new_weights)

            acc, loss = _eval_on_test(global_model, test_batched)
            elapsed = time.time() - t0
            result.rounds.append(RoundResult(acc, loss, div, elapsed))

            if (comm_round + 1) % 5 == 0 or comm_round == 0:
                log.info(
                    "[Non-IID %d/%d]  acc=%.4f  loss=%.4f  div=%.4f  time=%.1fs",
                    comm_round + 1, cfg.comms_round, acc, loss, div, elapsed,
                )

            # Early-stopping checks
            if cfg.target_accuracy is not None and acc >= cfg.target_accuracy:
                stop_reason = f"target accuracy {cfg.target_accuracy:.4f} reached"
                break

            elapsed_min = (time.time() - overall_start) / 60.0
            if cfg.max_total_minutes is not None and elapsed_min >= cfg.max_total_minutes:
                stop_reason = f"time budget {cfg.max_total_minutes:.1f} min reached"
                break

    except KeyboardInterrupt:
        stop_reason = "interrupted by user"

    if stop_reason:
        log.info("Non-IID stopped early: %s", stop_reason)

    if result.rounds:
        log.info(
            "Non-IID complete — rounds=%d  final acc=%.4f  loss=%.4f",
            len(result.rounds), result.accuracies[-1], result.losses[-1],
        )
    else:
        log.warning("Non-IID: no rounds were completed")

    return result
