"""
generate_figures.py
====================
IEEE-publication-ready figure generation for:
  "Robust Federated Learning for Cross-Continental Diabetes Prediction"

Figures produced (saved to output/figures/):
  fig1_fl_convergence.png      — FL Convergence Curve (Loss + AUC over 50 rounds)
  fig2_multi_model_roc.png     — Multi-Model ROC Curve (4 models)
  fig3_pr_curve.png            — Precision-Recall Curve (4 models + baseline)
  fig4_confusion_matrices.png  — Side-by-side Confusion Matrices (XGBoost & FL)

Usage:
  1. Run centralized_training.ipynb first (saves output/models/*.pkl)
  2. Run FL training once (saves output/fl_training_history.json)
  3. Run: python 4_evaluation_and_paper/generate_figures.py
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
import torch

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Root-relative paths (script lives in 4_evaluation_and_paper/)
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
ROOT_DIR     = SCRIPT_DIR.parent
OUTPUT_DIR   = ROOT_DIR / "output" / "figures"
MODEL_DIR    = ROOT_DIR / "output" / "models"
HISTORY_PATH = ROOT_DIR / "output" / "fl_training_history.json"
DATA_DIR     = ROOT_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# IEEE Visual Style Constants
# ---------------------------------------------------------------------------
matplotlib.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.dpi":        150,   # screen preview
    "savefig.dpi":       300,   # print quality
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#E0E0E0",
    "grid.linewidth":    0.6,
})

# IEEE-safe, B&W-distinguishable palette
COLORS      = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
LINE_STYLES = ["-", "--", ":", "-."]
MARKERS     = ["o", "s", "^", "D"]
MODEL_NAMES = ["XGBoost (Centralized)", "Logistic Regression", "Random Forest", "Federated PyTorch MLP"]

SAVE_KWARGS = dict(bbox_inches="tight", dpi=300)


# ===========================================================================
# DATA HELPERS
# ===========================================================================

def load_centralized_test_set():
    """
    Reconstruct the exact same test split the centralized notebook used.
    Uses Hospital_NY + Hospital_TX, StandardScaler from saved pkl.
    """
    ny_df = pd.read_csv(DATA_DIR / "Hospital_NY.csv")
    tx_df = pd.read_csv(DATA_DIR / "Hospital_TX.csv")
    df    = pd.concat([ny_df, tx_df], axis=0).reset_index(drop=True)

    X = df.drop(columns=["Diabetes_binary"]).values
    y = df["Diabetes_binary"].values

    # Reproduce the exact split (same seed) used in the notebook
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Apply the saved scaler (fit on train only — no leakage)
    scaler = joblib.load(MODEL_DIR / "centralized_scaler.pkl")
    X_test_scaled = scaler.transform(X_test)

    return X_test_scaled, y_test


def load_federated_model(input_dim=21):
    """Load the saved Federated PyTorch MLP weights."""
    # Import DiabetesMLP from the federated_learning folder
    fl_dir = str(ROOT_DIR / "3_federated_learning")
    if fl_dir not in sys.path:
        sys.path.insert(0, fl_dir)
    from model import DiabetesMLP  # noqa: E402

    net = DiabetesMLP(input_dim=input_dim)
    model_path = MODEL_DIR / "federated_mlp.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Federated model not found at {model_path}.\n"
            "Re-run the FL training (fl_server.py + both clients) — the server "
            "now saves the final model automatically after 50 rounds."
        )
    net.load_state_dict(torch.load(model_path, map_location="cpu"))
    net.eval()
    return net


def get_fl_probabilities(model, X_test):
    """Run inference on the FL MLP and return numpy probability array."""
    with torch.no_grad():
        tensor = torch.tensor(X_test, dtype=torch.float32)
        logits = model(tensor)
        probs  = torch.sigmoid(logits).squeeze(1).numpy()
    return probs


def load_all_probabilities(X_test, y_test):
    """
    Returns a list of (name, y_proba) tuples — one per model in MODEL_NAMES order.
    """
    proba_list = []

    # XGBoost
    xgb = joblib.load(MODEL_DIR / "xgboost.pkl")
    proba_list.append(("XGBoost (Centralized)", xgb.predict_proba(X_test)[:, 1]))

    # Logistic Regression
    lr  = joblib.load(MODEL_DIR / "logistic_regression.pkl")
    proba_list.append(("Logistic Regression", lr.predict_proba(X_test)[:, 1]))

    # Random Forest
    rf  = joblib.load(MODEL_DIR / "random_forest.pkl")
    proba_list.append(("Random Forest", rf.predict_proba(X_test)[:, 1]))

    # Federated PyTorch MLP (uses same test set scaled identically)
    fl_model = load_federated_model(input_dim=X_test.shape[1])
    proba_list.append(("Federated PyTorch MLP", get_fl_probabilities(fl_model, X_test)))

    return proba_list


# ===========================================================================
# FIGURE 1 — FL CONVERGENCE CURVE
# ===========================================================================

def plot_figure1_convergence():
    """
    Dual-axis line graph:  Loss (left Y) + AUC (right Y) over 50 FL rounds.
    Data source: output/fl_training_history.json
    """
    if not HISTORY_PATH.exists():
        print(f"[SKIP Fig 1] History file not found: {HISTORY_PATH}")
        print("  → Re-run the FL training once to generate it.")
        return

    with open(HISTORY_PATH) as f:
        hist = json.load(f)

    rounds = hist["rounds"]
    losses = hist["loss"]
    aucs   = hist["auc"]

    fig, ax1 = plt.subplots(figsize=(7, 4))

    color_loss = "#1f77b4"
    color_auc  = "#d62728"

    # --- Loss (left axis) ---
    ax1.set_xlabel("Communication Round")
    ax1.set_ylabel("Macro-Averaged BCE Loss", color=color_loss)
    ln1 = ax1.plot(rounds, losses, color=color_loss, linestyle="-",
                   linewidth=2, marker="o", markersize=3, markevery=5,
                   label="Loss (left axis)")
    ax1.tick_params(axis="y", labelcolor=color_loss)
    ax1.set_xlim(left=1)

    # --- AUC (right axis) ---
    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.set_ylabel("Macro-Averaged ROC-AUC", color=color_auc)
    ln2 = ax2.plot(rounds, aucs, color=color_auc, linestyle="--",
                   linewidth=2, marker="s", markersize=3, markevery=5,
                   label="AUC (right axis)")
    ax2.tick_params(axis="y", labelcolor=color_auc)
    ax2.set_ylim(0.5, 1.0)

    # Combine legend entries from both axes
    lns  = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="center right", framealpha=0.9)

    ax1.set_title("Figure 1: FL Convergence — FedProx over 50 Rounds (NY+TX Hospitals)")
    fig.tight_layout()
    out = OUTPUT_DIR / "fig1_fl_convergence.png"
    fig.savefig(out, **SAVE_KWARGS)
    plt.close(fig)
    print(f"✅ Fig 1 saved → {out}")


# ===========================================================================
# FIGURE 2 — MULTI-MODEL ROC CURVE
# ===========================================================================

def plot_figure2_roc(proba_list, y_test):
    """
    Single ROC chart comparing 4 models.
    Each model gets a distinct color + line style (IEEE B&W-safe).
    """
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    for i, (name, proba) in enumerate(proba_list):
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr,
                color=COLORS[i], linestyle=LINE_STYLES[i], linewidth=2,
                label=f"{name}  (AUC = {roc_auc:.3f})")

    # Diagonal chance line
    ax.plot([0, 1], [0, 1], color="#888888", linestyle="--",
            linewidth=1, label="Random Classifier (AUC = 0.500)")

    ax.set_xlabel("False Positive Rate (1 − Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
    ax.set_title("Figure 2: Multi-Model ROC Curve Comparison")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])

    fig.tight_layout()
    out = OUTPUT_DIR / "fig2_multi_model_roc.png"
    fig.savefig(out, **SAVE_KWARGS)
    plt.close(fig)
    print(f"✅ Fig 2 saved → {out}")


# ===========================================================================
# FIGURE 3 — PRECISION-RECALL CURVE
# ===========================================================================

def plot_figure3_pr(proba_list, y_test):
    """
    PR Curve for 4 models + dotted horizontal baseline (minority class prior).
    Baseline = P(y=1) in the test set ≈ 0.14 for this dataset.
    """
    minority_prior = float(y_test.mean())

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    for i, (name, proba) in enumerate(proba_list):
        precision, recall, _ = precision_recall_curve(y_test, proba)
        ap = average_precision_score(y_test, proba)
        ax.plot(recall, precision,
                color=COLORS[i], linestyle=LINE_STYLES[i], linewidth=2,
                label=f"{name}  (AP = {ap:.3f})")

    # Random-guessing baseline (flat line at minority class proportion)
    ax.axhline(y=minority_prior, color="#888888", linestyle=":",
               linewidth=1.5,
               label=f"Random Baseline  (P = {minority_prior:.2f})")

    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (Positive Predictive Value)")
    ax.set_title("Figure 3: Precision-Recall Curve Comparison")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])

    fig.tight_layout()
    out = OUTPUT_DIR / "fig3_pr_curve.png"
    fig.savefig(out, **SAVE_KWARGS)
    plt.close(fig)
    print(f"✅ Fig 3 saved → {out}")


# ===========================================================================
# FIGURE 4 — CONFUSION MATRICES (XGBoost vs FL)
# ===========================================================================

def _draw_cm(ax, cm, title, class_names=("Healthy (0)", "Diabetic (1)")):
    """
    Draw a single annotated confusion matrix heatmap with raw counts + %.
    """
    total   = cm.sum()
    cm_pct  = cm / total * 100

    # Build annotation: "N\n(X.X%)"
    annot = np.empty_like(cm, dtype=object)
    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            annot[r, c] = f"{cm[r, c]:,}\n({cm_pct[r, c]:.1f}%)"

    sns.heatmap(
        cm, annot=annot, fmt="", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, linecolor="#cccccc",
        cbar_kws={"shrink": 0.75},
        ax=ax,
    )
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label", fontsize=10)
    ax.tick_params(axis="both", labelsize=9)


def plot_figure4_confusion(proba_list, y_test, threshold=0.5):
    """
    1×2 subplot: Left = XGBoost (centralized), Right = Federated PyTorch MLP.
    """
    # Threshold probabilities
    preds_xgb = (proba_list[0][1] >= threshold).astype(int)  # XGBoost is index 0
    preds_fl  = (proba_list[3][1] >= threshold).astype(int)  # FL is index 3

    cm_xgb = confusion_matrix(y_test, preds_xgb)
    cm_fl  = confusion_matrix(y_test, preds_fl)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Figure 4: Confusion Matrices — Centralized vs. Federated",
                 fontsize=12, y=1.01)

    _draw_cm(axes[0], cm_xgb, "XGBoost (Centralized — All Data Pooled)")
    _draw_cm(axes[1], cm_fl,  "FedProx MLP (Federated — Privacy-Preserving)")

    fig.tight_layout()
    out = OUTPUT_DIR / "fig4_confusion_matrices.png"
    fig.savefig(out, **SAVE_KWARGS)
    plt.close(fig)
    print(f"✅ Fig 4 saved → {out}")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 60)
    print("  FL Research — IEEE Figure Generation")
    print("=" * 60)

    # --- Figure 1: only needs history JSON, no models ---
    print("\n[1/4] Generating FL Convergence Curve...")
    plot_figure1_convergence()

    # --- Load shared test set + model probabilities ---
    print("\n[2–4] Loading centralized test set and models...")
    try:
        X_test, y_test = load_centralized_test_set()
        proba_list = load_all_probabilities(X_test, y_test)
    except FileNotFoundError as e:
        print(f"\n⚠️  Could not load models: {e}")
        print("   → Run centralized notebook + FL training, then retry.")
        sys.exit(1)

    print(f"  Test set: {len(y_test):,} samples | "
          f"Diabetic rate: {y_test.mean():.1%}")

    # --- Figure 2: ROC ---
    print("\n[2/4] Generating Multi-Model ROC Curve...")
    plot_figure2_roc(proba_list, y_test)

    # --- Figure 3: PR ---
    print("\n[3/4] Generating Precision-Recall Curve...")
    plot_figure3_pr(proba_list, y_test)

    # --- Figure 4: Confusion Matrices ---
    print("\n[4/4] Generating Confusion Matrices...")
    plot_figure4_confusion(proba_list, y_test)

    print(f"\n{'='*60}")
    print(f"  All figures saved to: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
