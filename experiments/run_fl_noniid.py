import os
import subprocess
import sys
from pathlib import Path

def main():
    print("="*60)
    print("  LAUNCHING FEDERATED LEARNING EXPERIMENT: NON-IID MODE")
    print("="*60)
    
    # Force environment variable to noniid
    env = os.environ.copy()
    env["FL_DATA_MODE"] = "noniid"

    script_path = Path(__file__).resolve().parent.parent / "3_federated_learning" / "run_federated_experiment.py"
    
    # Passthrough any arguments (like --runs N)
    cmd = [sys.executable, str(script_path)] + sys.argv[1:]
    
    subprocess.run(cmd, env=env)

if __name__ == "__main__":
    main()
