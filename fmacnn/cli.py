"""Command-line interface entry point."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import Config, load_config
from .data import (
    batch_data,
    build_test_dataset,
    create_clients,
    load_dataset,
    split_dataset,
)
from .model import build_macnn
from .train import TrainResult, train_iid, train_noniid
from .visualize import plot_comparison, plot_single_run


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging(level: str, log_file: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fmacnn",
        description="Train FMACNN with federated learning (IID / Non-IID)",
    )
    p.add_argument(
        "--config", default="configs/default.yaml",
        help="Path to YAML config file (default: configs/default.yaml)",
    )
    p.add_argument(
        "--mode", choices=["iid", "noniid", "both"],
        help="Training mode; overrides the value in the config file",
    )
    p.add_argument(
        "--data-root",
        help="Dataset root directory; overrides config",
    )
    p.add_argument(
        "--output-dir",
        help="Output directory for models, metrics, and plots; overrides config",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    p.add_argument(
        "--debug", action="store_true",
        help="Enable debug mode (verbose per-client logging)",
    )
    return p


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _save_metrics(result: TrainResult, path: Path) -> None:
    payload = {
        "num_rounds": len(result.rounds),
        "accuracies": result.accuracies,
        "losses": result.losses,
        "divergences": result.divergences,
        "per_round_times_seconds": result.per_round_times,
    }
    path.write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    # Load and patch config
    config: Config = load_config(args.config)
    if args.mode:
        config.mode = args.mode
    if args.data_root:
        config.data.root = args.data_root
    if args.output_dir:
        config.output.dir = args.output_dir
    if args.debug:
        config.debug = True

    output_dir = Path(config.output.dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _setup_logging(args.log_level, log_file=output_dir / "train.log")
    log = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    log.info("Loading dataset from '%s'", config.data.root)
    images, labels, classes = load_dataset(
        config.data.root,
        target_size=tuple(config.data.target_size),
        color_mode=config.data.color_mode,
    )
    log.info(
        "Dataset loaded: %d samples  |  shape=%s  |  classes=%s",
        len(images), images.shape[1:], classes,
    )

    X_train, X_test, y_train, y_test = split_dataset(
        images, labels,
        test_size=config.data.test_size,
        random_state=config.data.random_state,
    )
    log.info("Split: train=%d  test=%d", len(X_train), len(X_test))
    test_batched = build_test_dataset(X_test, y_test)

    input_shape = images.shape[1:]
    num_classes = labels.shape[1]

    results: dict[str, TrainResult] = {}

    # ------------------------------------------------------------------
    # IID run
    # ------------------------------------------------------------------
    if config.mode in ("iid", "both"):
        log.info("Creating IID client partitions (%d clients)", config.iid.num_clients)
        iid_clients = create_clients(
            X_train, y_train,
            num_clients=config.iid.num_clients,
            iid=True,
        )
        iid_sizes = {name: len(shard) for name, shard in iid_clients.items()}
        iid_batched = {
            name: batch_data(shard, config.iid.batch_size)
            for name, shard in iid_clients.items()
        }

        global_model_iid = build_macnn(input_shape, num_classes, config.model.l2_reg)
        iid_result = train_iid(
            global_model_iid, iid_batched, iid_sizes,
            test_batched, config, input_shape, num_classes,
        )
        results["iid"] = iid_result

        model_path = output_dir / "model_iid.keras"
        global_model_iid.save(model_path)
        log.info("IID model saved → %s", model_path)

        plot_single_run(iid_result, "IID", output_dir / "fl_iid_analysis.png")
        _save_metrics(iid_result, output_dir / "metrics_iid.json")
        log.info("IID plots and metrics saved to %s", output_dir)

    # ------------------------------------------------------------------
    # Non-IID run
    # ------------------------------------------------------------------
    if config.mode in ("noniid", "both"):
        log.info("Creating Non-IID client partitions (%d clients)", config.noniid.num_clients)
        noniid_clients = create_clients(
            X_train, y_train,
            num_clients=config.noniid.num_clients,
            iid=False,
        )
        noniid_sizes = {name: len(shard) for name, shard in noniid_clients.items()}
        noniid_batched = {
            name: batch_data(shard, config.noniid.batch_size)
            for name, shard in noniid_clients.items()
        }

        global_model_noniid = build_macnn(input_shape, num_classes, config.model.l2_reg)
        noniid_result = train_noniid(
            global_model_noniid, noniid_batched, noniid_sizes,
            test_batched, config, input_shape, num_classes,
        )
        results["noniid"] = noniid_result

        model_path = output_dir / "model_noniid.keras"
        global_model_noniid.save(model_path)
        log.info("Non-IID model saved → %s", model_path)

        plot_single_run(noniid_result, "Non-IID", output_dir / "fl_noniid_analysis.png")
        _save_metrics(noniid_result, output_dir / "metrics_noniid.json")
        log.info("Non-IID plots and metrics saved to %s", output_dir)

    # ------------------------------------------------------------------
    # Comparison plot (only when both modes ran)
    # ------------------------------------------------------------------
    if "iid" in results and "noniid" in results:
        plot_comparison(results["iid"], results["noniid"], output_dir / "fl_comparison.png")
        log.info("Comparison plot saved → %s", output_dir / "fl_comparison.png")

    log.info("All done. Outputs in: %s", output_dir)
