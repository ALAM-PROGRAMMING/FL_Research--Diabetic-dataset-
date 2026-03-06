import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
EXP_DIR  = ROOT_DIR / "output" / "experiment_runs"
FIG_DIR  = ROOT_DIR / "output" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

def plot_convergence_comparison():
    print("="*60)
    print("  FL CONVERGENCE COMPARISON (IID vs NON-IID)")
    print("="*60)
    
    iid_summary_path = EXP_DIR / "iid" / "experiment_summary.json"
    noniid_summary_path = EXP_DIR / "noniid" / "experiment_summary.json"
    
    if not iid_summary_path.exists() or not noniid_summary_path.exists():
        print("Missing summary JSON files. Please verify that BOTH experiments have been run:")
        print("  python experiments/run_fl_iid.py")
        print("  python experiments/run_fl_noniid.py")
        return

    with open(iid_summary_path) as f:
        iid_data = json.load(f)
    with open(noniid_summary_path) as f:
        noniid_data = json.load(f)

    # -------------------------------------------------------------
    # 1. Generate Terminal Table
    # -------------------------------------------------------------
    table = []
    
    def format_stat(d, key):
        if key in d and d[key] and d[key]["mean"] is not None:
            return f"{d[key]['mean']:.4f} ± {d[key]['stdev']:.4f}"
        return "N/A"

    table.append({
        "Mode": "IID",
        "Final AUC": format_stat(iid_data, "final_auc"),
        "Final Loss": format_stat(iid_data, "final_loss"),
        "Final Recall": format_stat(iid_data, "final_recall"),
        "Final F1": format_stat(iid_data, "final_f1")
    })
    
    table.append({
        "Mode": "Non-IID",
        "Final AUC": format_stat(noniid_data, "final_auc"),
        "Final Loss": format_stat(noniid_data, "final_loss"),
        "Final Recall": format_stat(noniid_data, "final_recall"),
        "Final F1": format_stat(noniid_data, "final_f1")
    })
    
    df = pd.DataFrame(table)
    print("\n" + df.to_string(index=False) + "\n")

    # -------------------------------------------------------------
    # 2. Extract Training Curves
    # -------------------------------------------------------------
    # To plot full curves, we need one of the run files from each dir
    iid_run = list((EXP_DIR / "iid").glob("run_*_metrics.json"))
    noniid_run = list((EXP_DIR / "noniid").glob("run_*_metrics.json"))
    
    if not iid_run or not noniid_run:
        print("Missing historic run metrics needed to plot curves.")
        return
        
    with open(iid_run[0]) as f:
        iid_hist = json.load(f)
    with open(noniid_run[0]) as f:
        noniid_hist = json.load(f)
        
    rounds = iid_hist.get("rounds", list(range(1, len(iid_hist.get("auc", [])) + 1)))
    
    # -------------------------------------------------------------
    # 3. Generate Convergence Figure
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # AUC Plot
    if "auc" in iid_hist and "auc" in noniid_hist:
        ax1.plot(rounds, iid_hist["auc"], label="IID Cohort", color="#4A90E2", linewidth=2.5)
        ax1.plot(rounds, noniid_hist["auc"], label="Non-IID Cohort", color="#F5A623", linewidth=2.5, linestyle="--")
        ax1.set_title("Federated Training Convergence (Global AUC)", fontsize=13, fontweight='bold', pad=15)
        ax1.set_xlabel("Communication Round", fontsize=11)
        ax1.set_ylabel("Area Under ROC Curve (AUC)", fontsize=11)
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend(loc="lower right")
    
    # Loss Plot
    if "loss" in iid_hist and "loss" in noniid_hist:
        ax2.plot(rounds[:len(iid_hist["loss"])], iid_hist["loss"], label="IID Cohort", color="#4A90E2", linewidth=2.5)
        ax2.plot(rounds[:len(noniid_hist["loss"])], noniid_hist["loss"], label="Non-IID Cohort", color="#F5A623", linewidth=2.5, linestyle="--")
        ax2.set_title("Federated Training Loss Minimization", fontsize=13, fontweight='bold', pad=15)
        ax2.set_xlabel("Communication Round", fontsize=11)
        ax2.set_ylabel("Global Aggregated Loss", fontsize=11)
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend(loc="upper right")
        
    plt.tight_layout()
    out_path = FIG_DIR / "fig_convergence_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Figure saved to: {out_path}")

if __name__ == "__main__":
    plot_convergence_comparison()
