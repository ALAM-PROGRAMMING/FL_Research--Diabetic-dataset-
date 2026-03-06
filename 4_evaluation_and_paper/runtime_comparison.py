import subprocess
import time
import sys
import json
import matplotlib.pyplot as plt
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = ROOT_DIR / "output" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
PYTHON = sys.executable

def extract_notebook_code(notebook_path: Path, output_py_path: Path):
    """Extract code from a Jupyter Notebook to a temporary Python script."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    code_cells = [cell["source"] for cell in nb["cells"] if cell["cell_type"] == "code"]
    
    with open(output_py_path, "w", encoding="utf-8") as f:
        for lines in code_cells:
            f.write("".join(lines) + "\n\n")

def measure_runtime():
    if len(sys.argv) < 2 or sys.argv[1] != "--execute":
        print("This script is designed to measure training time by re-running the full training pipelines.")
        print("To actually execute the runtime comparison, run:")
        print(f"    python {Path(__file__).name} --execute")
        return

    print("="*60)
    print("  RUNTIME COMPARISON MEASUREMENT")
    print("="*60)
    
    runtimes = {}
    
    # -------------------------------------------------------------
    # 1. Centralized Training Runtime
    # -------------------------------------------------------------
    tmp_cent_script = ROOT_DIR / "2_baselines" / "tmp_run_cent.py"
    cent_nb         = ROOT_DIR / "2_baselines" / "centralized_training.ipynb"
    
    if cent_nb.exists():
        print("[1/3] Measuring Centralized Training Time...")
        extract_notebook_code(cent_nb, tmp_cent_script)
        
        t0 = time.perf_counter()
        subprocess.run([PYTHON, str(tmp_cent_script)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        t1 = time.perf_counter()
        
        runtimes["Centralized"] = t1 - t0
        print(f"      Time: {runtimes['Centralized']:.2f} seconds")
        
        if tmp_cent_script.exists():
            tmp_cent_script.unlink()
    else:
        print("Centralized notebook not found, skipping...")

    # -------------------------------------------------------------
    # 2. FL IID Runtime (1 run)
    # -------------------------------------------------------------
    fl_iid_script = ROOT_DIR / "experiments" / "run_fl_iid.py"
    if fl_iid_script.exists():
        print("[2/3] Measuring Federated Learning (IID) Training Time...")
        t0 = time.perf_counter()
        # Suppressing output so it doesn't flood the terminal
        subprocess.run([PYTHON, str(fl_iid_script), "--runs", "1"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        t1 = time.perf_counter()
        
        runtimes["FL (IID)"] = t1 - t0
        print(f"      Time: {runtimes['FL (IID)']:.2f} seconds")
    else:
        print("FL IID script not found, skipping...")

    # -------------------------------------------------------------
    # 3. FL Non-IID Runtime (1 run)
    # -------------------------------------------------------------
    fl_noniid_script = ROOT_DIR / "experiments" / "run_fl_noniid.py"
    if fl_noniid_script.exists():
        print("[3/3] Measuring Federated Learning (Non-IID) Training Time...")
        t0 = time.perf_counter()
        subprocess.run([PYTHON, str(fl_noniid_script), "--runs", "1"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        t1 = time.perf_counter()
        
        runtimes["FL (Non-IID)"] = t1 - t0
        print(f"      Time: {runtimes['FL (Non-IID)']:.2f} seconds")
    else:
        print("FL Non-IID script not found, skipping...")

    if not runtimes:
        print("No runtimes measured.")
        return

    # -------------------------------------------------------------
    # Generate Figure
    # -------------------------------------------------------------
    labels = list(runtimes.keys())
    times  = list(runtimes.values())
    
    plt.figure(figsize=(8, 5))
    colors = ['#4A90E2', '#50E3C2', '#F5A623']
    
    bars = plt.barh(labels, times, color=colors[:len(labels)], edgecolor='black')
    
    plt.title("Runtime Comparison: Centralized vs Federated Learning", pad=20, fontsize=14, fontweight='bold')
    plt.xlabel("Total Wall-Clock Time (seconds)", fontsize=12)
    
    # Add text to bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + (max(times)*0.02), bar.get_y() + bar.get_height()/2, 
                 f"{width:.1f}s", ha='left', va='center', fontweight='bold')
                 
    # Adjust layout to fit text
    plt.xlim(0, max(times) * 1.15)
    
    plt.tight_layout()
    out_path = FIGURES_DIR / "fig_runtime_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Figure saved to: {out_path}")

if __name__ == "__main__":
    measure_runtime()
