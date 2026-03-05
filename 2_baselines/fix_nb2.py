import json

with open("02_centralized_baselines.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        for i, line in enumerate(cell["source"]):
            if "plt.grid(True, linestyle=" in line:
                cell["source"][i] = "plt.grid(True, linestyle='--', alpha=0.5)\n"

with open("02_centralized_baselines.ipynb", "w") as f:
    json.dump(nb, f, indent=2)
