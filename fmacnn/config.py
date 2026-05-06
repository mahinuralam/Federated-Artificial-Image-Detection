"""Hierarchical configuration dataclasses with YAML loading."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


# ---------------------------------------------------------------------------
# Section dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    root: str = "Datasets/RealAIGI"
    target_size: tuple = (256, 256)
    color_mode: str = "rgb"
    test_size: float = 0.1
    random_state: int = 42


@dataclass
class ModelConfig:
    l2_reg: float = 1e-4


@dataclass
class FederatedConfig:
    """Shared federated-aggregation settings."""
    threshold_percentile: int = 95
    momentum: float = 0.9
    nesterov: bool = True


@dataclass
class IIDConfig:
    num_clients: int = 100
    num_clients_per_round: int = 10
    comms_round: int = 100
    local_epochs: int = 3
    batch_size: int = 32
    initial_lr: float = 0.01
    lr_decay_rate: float = 0.96
    lr_decay_steps: int = 10   # round interval between LR decay steps


@dataclass
class NonIIDConfig:
    num_clients: int = 100
    num_clients_per_round: int = 10
    comms_round: int = 150
    local_epochs: int = 3
    batch_size: int = 32
    initial_lr: float = 0.008
    lr_decay_rate: float = 0.98
    lr_decay_steps: int = 15
    max_batches_per_client: Optional[int] = None
    max_total_minutes: Optional[float] = None
    target_accuracy: Optional[float] = None


@dataclass
class OutputConfig:
    dir: str = "outputs"


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    iid: IIDConfig = field(default_factory=IIDConfig)
    noniid: NonIIDConfig = field(default_factory=NonIIDConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    mode: str = "both"   # "iid" | "noniid" | "both"
    debug: bool = False


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _apply_dict(instance: object, updates: dict) -> None:
    """Recursively overwrite dataclass fields from a plain dict."""
    for f in dataclasses.fields(instance):
        if f.name not in updates:
            continue
        val = updates[f.name]
        current = getattr(instance, f.name)
        if dataclasses.is_dataclass(current) and isinstance(val, dict):
            _apply_dict(current, val)
        else:
            if isinstance(current, tuple) and isinstance(val, list):
                val = tuple(val)
            setattr(instance, f.name, val)


def load_config(path: str | Path) -> Config:
    """Load a YAML file and merge it on top of the default Config."""
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    cfg = Config()
    _apply_dict(cfg, data)
    return cfg
