"""Training result visualisations."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .train import TrainResult

sns.set_style("whitegrid")

_WINDOW = 10  # rounds for moving-average smoothing


def _moving_avg(data: list[float], window: int = _WINDOW) -> np.ndarray:
    if len(data) < window:
        return np.array(data)
    return np.convolve(data, np.ones(window) / window, mode="valid")


def plot_single_run(
    result: TrainResult,
    title: str,
    output_path: str | Path,
) -> None:
    """9-panel figure: accuracy, loss, divergence, moving avg, deltas, time, stats, histogram."""
    acc = result.accuracies
    loss = result.losses
    divs = result.divergences
    times = result.per_round_times
    rounds = range(len(acc))
    total_time = sum(times)

    fig, axes = plt.subplots(3, 3, figsize=(20, 12))
    axes = axes.flatten()

    # 1. Accuracy
    axes[0].plot(rounds, acc, "b-", linewidth=2)
    axes[0].fill_between(rounds, acc, alpha=0.3)
    axes[0].set(xlabel="Round", ylabel="Accuracy", title=f"Accuracy ({title})")

    # 2. Loss
    axes[1].plot(rounds, loss, "r-", linewidth=2)
    axes[1].fill_between(rounds, loss, alpha=0.3, color="red")
    axes[1].set(xlabel="Round", ylabel="Loss", title=f"Loss ({title})")

    # 3. Weight divergence
    axes[2].plot(rounds, divs, "g-", linewidth=2)
    axes[2].set(xlabel="Round", ylabel="L2 Divergence", title=f"Weight Divergence ({title})")

    # 4. Smoothed accuracy
    ma = _moving_avg(acc)
    axes[3].plot(range(len(ma)), ma, "purple", linewidth=2, label=f"MA({_WINDOW})")
    axes[3].set(xlabel="Round", ylabel="Accuracy", title=f"Smoothed Accuracy ({title})")
    axes[3].legend()

    # 5. Accuracy delta
    if len(acc) > 1:
        axes[4].plot(range(len(acc) - 1), np.diff(acc), "orange", linewidth=2)
        axes[4].axhline(0, color="black", linestyle="--", alpha=0.5)
    axes[4].set(xlabel="Round", ylabel="Δ Accuracy", title=f"Accuracy Change ({title})")

    # 6. Loss delta
    if len(loss) > 1:
        axes[5].plot(range(len(loss) - 1), np.diff(loss), "brown", linewidth=2)
        axes[5].axhline(0, color="black", linestyle="--", alpha=0.5)
    axes[5].set(xlabel="Round", ylabel="Δ Loss", title=f"Loss Change ({title})")

    # 7. Cumulative wall-clock time
    axes[6].plot(rounds, np.cumsum(times), "cyan", linewidth=2)
    axes[6].set(xlabel="Round", ylabel="Seconds", title=f"Cumulative Time ({title})")

    # 8. Stats text box
    axes[7].axis("off")
    stats = (
        f"STATISTICS — {title}\n"
        f"{'─' * 32}\n"
        f"Rounds : {len(acc)}\n"
        f"Acc    : init={acc[0]:.4f}  final={acc[-1]:.4f}\n"
        f"         max={max(acc):.4f}   mean={np.mean(acc):.4f}\n"
        f"Loss   : final={loss[-1]:.4f}  min={min(loss):.4f}\n"
        f"Time   : total={total_time:.1f}s  per_round={total_time / len(acc):.1f}s\n"
        f"Div    : mean={np.mean(divs):.4f}  max={max(divs):.4f}"
    )
    axes[7].text(
        0.05, 0.5, stats, fontsize=11, family="monospace",
        va="center", transform=axes[7].transAxes,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    # 9. Accuracy histogram
    axes[8].hist(acc, bins=30, edgecolor="black", alpha=0.7, color="skyblue")
    axes[8].axvline(np.mean(acc), color="red", linestyle="--", linewidth=2, label="Mean")
    axes[8].axvline(np.median(acc), color="green", linestyle="--", linewidth=2, label="Median")
    axes[8].set(xlabel="Accuracy", ylabel="Frequency", title=f"Accuracy Distribution ({title})")
    axes[8].legend()

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(
    iid: TrainResult,
    noniid: TrainResult,
    output_path: str | Path,
) -> None:
    """6-panel IID vs Non-IID comparison figure."""
    iid_acc, noniid_acc = iid.accuracies, noniid.accuracies
    iid_loss, noniid_loss = iid.losses, noniid.losses
    iid_div, noniid_div = iid.divergences, noniid.divergences

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Accuracy
    axes[0, 0].plot(iid_acc, "b-", linewidth=2, label="IID", alpha=0.8)
    axes[0, 0].plot(noniid_acc, "r-", linewidth=2, label="Non-IID", alpha=0.8)
    axes[0, 0].set(xlabel="Round", ylabel="Accuracy", title="Accuracy: IID vs Non-IID")
    axes[0, 0].legend()

    # Loss
    axes[0, 1].plot(iid_loss, "b-", linewidth=2, label="IID", alpha=0.8)
    axes[0, 1].plot(noniid_loss, "r-", linewidth=2, label="Non-IID", alpha=0.8)
    axes[0, 1].set(xlabel="Round", ylabel="Loss", title="Loss: IID vs Non-IID")
    axes[0, 1].legend()

    # Divergence
    axes[0, 2].plot(iid_div, "b-", linewidth=2, label="IID", alpha=0.8)
    axes[0, 2].plot(noniid_div, "r-", linewidth=2, label="Non-IID", alpha=0.8)
    axes[0, 2].set(xlabel="Round", ylabel="L2 Divergence", title="Weight Divergence: IID vs Non-IID")
    axes[0, 2].legend()

    # Accuracy distribution overlay
    axes[1, 0].hist(iid_acc, bins=20, alpha=0.6, label="IID", color="blue", edgecolor="black")
    axes[1, 0].hist(noniid_acc, bins=20, alpha=0.6, label="Non-IID", color="red", edgecolor="black")
    axes[1, 0].set(xlabel="Accuracy", ylabel="Frequency", title="Accuracy Distribution")
    axes[1, 0].legend()

    # Box plot
    axes[1, 1].boxplot(
        [iid_acc, noniid_acc],
        labels=["IID", "Non-IID"],
        patch_artist=True,
        boxprops=dict(facecolor="lightblue", alpha=0.7),
        medianprops=dict(color="red", linewidth=2),
    )
    axes[1, 1].set(ylabel="Accuracy", title="Accuracy Spread")

    # Summary stats
    axes[1, 2].axis("off")
    gap_acc = abs(iid_acc[-1] - noniid_acc[-1])
    gap_loss = abs(iid_loss[-1] - noniid_loss[-1])
    text = (
        f"COMPARATIVE SUMMARY\n{'═' * 32}\n"
        f"IID     — final={iid_acc[-1]:.4f}  max={max(iid_acc):.4f}\n"
        f"Non-IID — final={noniid_acc[-1]:.4f}  max={max(noniid_acc):.4f}\n\n"
        f"Accuracy gap : {gap_acc:.4f}\n"
        f"Loss gap     : {gap_loss:.4f}\n\n"
        f"Rounds — IID:{len(iid_acc)}  Non-IID:{len(noniid_acc)}"
    )
    axes[1, 2].text(
        0.1, 0.5, text, fontsize=11, family="monospace",
        va="center", transform=axes[1, 2].transAxes,
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.3),
    )

    for row in axes:
        for ax in row:
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
