"""
communication_analysis.py
===========================
Computes the communication cost of the federated learning training process
and compares it against a hypothetical centralized baseline (raw data transfer).

Output:
    Console: formatted metrics table
    File:    output/figures/fig_communication_cost.png

This analysis supports the paper's privacy-utility argument:
FL achieves near-centralized AUC while transmitting ONLY gradient summaries —
never raw patient data.
"""

import os
import sys
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from collections import OrderedDict

matplotlib.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "axes.grid":         True,
    "grid.color":        "#E0E0E0",
    "grid.linewidth":    0.6,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR   = SCRIPT_DIR.parent
FIG_DIR    = ROOT_DIR / "output" / "figures"
DATA_DIR   = ROOT_DIR / "data" / "processed"
FL_DIR     = ROOT_DIR / "3_federated_learning"
FIG_DIR.mkdir(parents=True, exist_ok=True)

if str(FL_DIR) not in sys.path:
    sys.path.insert(0, str(FL_DIR))


# ---------------------------------------------------------------------------
# 1. Compute FL Model Size
# ---------------------------------------------------------------------------

def compute_model_size_mb(input_dim: int = 21) -> float:
    """
    Compute DiabetesMLP parameter count and size in MB.
    Uses float32 (4 bytes per parameter) — the PyTorch default.
    Does NOT load a checkpoint; reconstructs from architecture definition.
    """
    from model import DiabetesMLP  # noqa: E402
    import torch
    model = DiabetesMLP(input_dim=input_dim)
    total_params = sum(p.numel() for p in model.parameters())
    size_mb = (total_params * 4) / (1024 ** 2)   # float32 = 4 bytes
    return total_params, size_mb


# ---------------------------------------------------------------------------
# 2. Compute Raw Dataset Size
# ---------------------------------------------------------------------------

def compute_dataset_size_mb() -> dict:
    """
    Measure the on-disk size of each hospital CSV (the data that would need
    to be transferred to a central server in the non-FL scenario).
    """
    result = {}
    for fname in ["Hospital_NY.csv", "Hospital_TX.csv"]:
        path = DATA_DIR / fname
        if path.exists():
            result[fname] = path.stat().st_size / (1024 ** 2)
    return result


# ---------------------------------------------------------------------------
# 3. Print Report
# ---------------------------------------------------------------------------

def print_report(metrics: dict) -> None:
    print("\n" + "=" * 60)
    print("  FL COMMUNICATION EFFICIENCY ANALYSIS")
    print("=" * 60)

    print("\n── Model Architecture ────────────────────────────────────")
    print(f"  Parameters      : {metrics['model_params']:,}")
    print(f"  Model size      : {metrics['model_size_mb']:.4f} MB")

    print("\n── FL Communication Cost ─────────────────────────────────")
    print(f"  Clients         : {metrics['num_clients']}")
    print(f"  Rounds          : {metrics['num_rounds']}")
    print(f"  Directions      : 2  (server → client download + client → server upload)")
    print(f"  Per-round TX    : {metrics['per_round_mb']:.4f} MB")
    print(f"  TOTAL FL TX     : {metrics['total_fl_mb']:.4f} MB")

    print("\n── Centralized Baseline (Raw Data Transfer) ──────────────")
    for fname, size_mb in metrics['dataset_sizes_mb'].items():
        print(f"  {fname:<25}: {size_mb:.2f} MB")
    print(f"  TOTAL Centralized : {metrics['total_centralized_mb']:.2f} MB")

    print("\n── Savings ───────────────────────────────────────────────")
    print(f"  Data saved      : {metrics['savings_mb']:.2f} MB")
    print(f"  Reduction ratio : {metrics['savings_pct']:.1f}%")
    print(f"  FL transmits only {(metrics['total_fl_mb'] / metrics['total_centralized_mb'] * 100):.2f}%"
          " of raw data volume\n")


# ---------------------------------------------------------------------------
# 4. Plot
# ---------------------------------------------------------------------------

def plot_communication_cost(metrics: dict) -> None:
    categories = [
        "Centralized\n(Raw Data Transfer)",
        "Federated Learning\n(FedProx, 50 Rounds)",
    ]
    values = [metrics["total_centralized_mb"], metrics["total_fl_mb"]]
    colors = ["#d62728", "#1f77b4"]

    fig, ax = plt.subplots(figsize=(6.5, 5))

    bars = ax.bar(categories, values, color=colors, width=0.5,
                  edgecolor="white", linewidth=1.2)

    # Annotate bars with values
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.2f} MB",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    # Savings annotation
    ax.annotate(
        f"{metrics['savings_pct']:.1f}% reduction",
        xy=(1, values[1]),
        xytext=(0.5, (values[0] + values[1]) / 2),
        textcoords="data",
        fontsize=9, color="#2ca02c", fontweight="bold",
        ha="center",
        arrowprops=dict(arrowstyle="<->", color="#2ca02c", lw=1.5),
    )

    ax.set_ylabel("Total Data Transmitted (MB)")
    ax.set_title(
        "Communication Cost: Centralized vs. Federated Learning\n"
        f"(FL: {metrics['num_clients']} clients × {metrics['num_rounds']} rounds × "
        f"2 directions × {metrics['model_size_mb']:.3f} MB/model)"
    )
    ax.set_ylim(0, values[0] * 1.25)

    fig.tight_layout()
    out = FIG_DIR / "fig_communication_cost.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Figure saved → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # 1. Model size
    total_params, model_size_mb = compute_model_size_mb(input_dim=21)

    # 2. Dataset sizes  
    dataset_sizes_mb = compute_dataset_size_mb()
    total_centralized_mb = sum(dataset_sizes_mb.values())

    # 3. FL cost
    num_clients  = 2   # NY + TX
    num_rounds   = 50
    directions   = 2   # upload + download
    per_round_mb = model_size_mb * num_clients * directions
    total_fl_mb  = per_round_mb * num_rounds

    # 4. Savings
    savings_mb  = total_centralized_mb - total_fl_mb
    savings_pct = (savings_mb / total_centralized_mb) * 100

    metrics = {
        "model_params":         total_params,
        "model_size_mb":        model_size_mb,
        "num_clients":          num_clients,
        "num_rounds":           num_rounds,
        "per_round_mb":         per_round_mb,
        "total_fl_mb":          total_fl_mb,
        "dataset_sizes_mb":     dataset_sizes_mb,
        "total_centralized_mb": total_centralized_mb,
        "savings_mb":           savings_mb,
        "savings_pct":          savings_pct,
    }

    print_report(metrics)
    plot_communication_cost(metrics)

    # Save metrics to JSON for reference
    metrics_copy = dict(metrics)
    metrics_copy.pop("dataset_sizes_mb")
    metrics_copy["dataset_sizes_mb"] = {k: round(v, 4) for k, v in dataset_sizes_mb.items()}
    out_json = FIG_DIR.parent / "communication_metrics.json"
    with open(out_json, "w") as f:
        json.dump(metrics_copy, f, indent=2)
    print(f"✅ Metrics JSON saved → {out_json}")


if __name__ == "__main__":
    main()
