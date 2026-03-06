"""
run_federated_experiment.py
============================
Orchestrates multiple runs of the full FL training pipeline to produce
statistically-significant metrics for the IEEE paper.

Architecture (zero logic duplication):
    subprocess → fl_server.py  (unmodified)
    subprocess → client_ny.py  (unmodified)
    subprocess → client_tx.py  (unmodified)

After each run, fl_server.py writes output/fl_training_history.json.
This script moves it to output/experiment_runs/run_N_metrics.json.

Usage:
    python run_federated_experiment.py             # 5 runs (default)
    python run_federated_experiment.py --runs 3    # 3 runs
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to 3_federated_learning/)
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
ROOT_DIR     = SCRIPT_DIR.parent
HISTORY_PATH = ROOT_DIR / "output" / "fl_training_history.json"
RUNS_DIR     = ROOT_DIR / "output" / "experiment_runs"
PYTHON       = sys.executable   # Use the same venv Python as the caller

SERVER_SCRIPT    = str(SCRIPT_DIR / "fl_server.py")
CLIENT_NY_SCRIPT = str(SCRIPT_DIR / "client_ny.py")
CLIENT_TX_SCRIPT = str(SCRIPT_DIR / "client_tx.py")

SERVER_STARTUP_WAIT = 3   # seconds to wait for the server to bind port 8080


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_one_experiment(run_number: int) -> Path:
    """
    Launches server + both clients as subprocesses, waits for completion,
    then archives the resulting history JSON.

    Returns the path of the saved run metrics file.
    """
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT RUN {run_number}")
    print(f"{'='*60}")

    # 1. Start the FL server
    server_proc = subprocess.Popen(
        [PYTHON, SERVER_SCRIPT],
        cwd=str(SCRIPT_DIR),
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    print(f"[Run {run_number}] Server started (PID {server_proc.pid}). "
          f"Waiting {SERVER_STARTUP_WAIT}s for it to bind port 8080...")
    time.sleep(SERVER_STARTUP_WAIT)

    # 2. Start both clients simultaneously
    client_ny_proc = subprocess.Popen(
        [PYTHON, CLIENT_NY_SCRIPT],
        cwd=str(SCRIPT_DIR),
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    client_tx_proc = subprocess.Popen(
        [PYTHON, CLIENT_TX_SCRIPT],
        cwd=str(SCRIPT_DIR),
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    print(f"[Run {run_number}] Client NY (PID {client_ny_proc.pid}) and "
          f"Client TX (PID {client_tx_proc.pid}) launched.")

    # 3. Wait for all three processes to finish
    server_proc.wait()
    client_ny_proc.wait()
    client_tx_proc.wait()

    if server_proc.returncode != 0:
        raise RuntimeError(
            f"[Run {run_number}] Server exited with code {server_proc.returncode}. "
            "Check fl_server.py output."
        )

    # 4. Verify history file was created
    if not HISTORY_PATH.exists():
        raise FileNotFoundError(
            f"[Run {run_number}] Expected history file not found: {HISTORY_PATH}\n"
            "Ensure fl_server.py is saving output/fl_training_history.json."
        )

    # 5. Archive to experiment_runs/run_N_metrics.json
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    dest = RUNS_DIR / f"run_{run_number}_metrics.json"
    shutil.move(str(HISTORY_PATH), str(dest))
    print(f"[Run {run_number}] ✅ Metrics saved → {dest}")
    return dest


def compute_summary(run_files: list) -> dict:
    """
    Aggregate AUC, Loss from all run files and compute mean ± std
    of the final-round value for each metric.
    """
    final_aucs  = []
    final_losses = []

    for path in run_files:
        with open(path) as f:
            data = json.load(f)
        if data.get("auc"):
            final_aucs.append(data["auc"][-1])
        if data.get("loss"):
            final_losses.append(data["loss"][-1])

    import statistics
    summary = {
        "num_runs":       len(run_files),
        "final_auc": {
            "values": final_aucs,
            "mean":   statistics.mean(final_aucs)   if final_aucs   else None,
            "stdev":  statistics.stdev(final_aucs)  if len(final_aucs) > 1  else 0,
        },
        "final_loss": {
            "values": final_losses,
            "mean":   statistics.mean(final_losses)  if final_losses  else None,
            "stdev":  statistics.stdev(final_losses) if len(final_losses) > 1 else 0,
        },
    }
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run the FL pipeline N times for statistical significance."
    )
    parser.add_argument(
        "--runs", type=int, default=5,
        help="Number of independent training runs (default: 5)."
    )
    args = parser.parse_args()

    print(f"\n🚀 Starting {args.runs} independent FL experiment runs.")
    print(f"   Output directory: {RUNS_DIR}")

    run_files = []
    for i in range(1, args.runs + 1):
        dest = run_one_experiment(run_number=i)
        run_files.append(dest)

    # Compute and save summary
    summary = compute_summary(run_files)
    summary_path = RUNS_DIR / "experiment_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  ALL {args.runs} RUNS COMPLETE")
    print(f"{'='*60}")
    print(f"  Final AUC  : {summary['final_auc']['mean']:.4f} "
          f"± {summary['final_auc']['stdev']:.4f}")
    print(f"  Final Loss : {summary['final_loss']['mean']:.4f} "
          f"± {summary['final_loss']['stdev']:.4f}")
    print(f"  Summary    : {summary_path}")


if __name__ == "__main__":
    main()
