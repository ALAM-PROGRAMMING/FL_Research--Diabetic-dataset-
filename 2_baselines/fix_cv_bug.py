import json
import os

path = "d:/DS PROJECTS/FL_Research/2_baselines/centralized_training.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find the markdown cell with VISUALS or add it if it doesn't exist
visuals_idx = -1
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "markdown":
        source = "".join(cell["source"])
        if "VISUALS" in source.upper():
            visuals_idx = i
            break

# If we couldn't find a VISUALS markdown cell, let's append one
if visuals_idx == -1:
    markdown_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## VISUALS\n", "Unified ROC Curve for all Centralized Models"]
    }
    nb["cells"].append(markdown_cell)
    visuals_idx = len(nb["cells"]) - 1

# Define our new robust code block
new_plot_code = [
    "# reconstructed pristine X_test & y_test to prevent CV loop bleed\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X, y, test_size=0.2, random_state=42, stratify=y\n",
    ")\n",
    "\n",
    "scaler = StandardScaler()\n",
    "X_train = scaler.fit_transform(X_train)\n",
    "X_test = scaler.transform(X_test)\n",
    "\n",
    "# Recalculate probabilities strictly on the true test set\n",
    "fresh_proba_lr = lr_model.predict_proba(X_test)[:, 1]\n",
    "fresh_proba_rf = rf_model.predict_proba(X_test)[:, 1]\n",
    "fresh_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]\n",
    "fresh_proba_vote = voting_model.predict_proba(X_test)[:, 1]\n",
    "\n",
    "# Get fresh auc scores\n",
    "fresh_auc_lr = roc_auc_score(y_test, fresh_proba_lr)\n",
    "fresh_auc_rf = roc_auc_score(y_test, fresh_proba_rf)\n",
    "fresh_auc_xgb = roc_auc_score(y_test, fresh_proba_xgb)\n",
    "fresh_auc_vote = roc_auc_score(y_test, fresh_proba_vote)\n",
    "\n",
    "fpr_lr, tpr_lr, _ = roc_curve(y_test, fresh_proba_lr)\n",
    "fpr_rf, tpr_rf, _ = roc_curve(y_test, fresh_proba_rf)\n",
    "fpr_xgb, tpr_xgb, _ = roc_curve(y_test, fresh_proba_xgb)\n",
    "fpr_vote, tpr_vote, _ = roc_curve(y_test, fresh_proba_vote)\n",
    "\n",
    "plt.figure(figsize=(8, 6))\n",
    "plt.style.use(\"default\")\n",
    "\n",
    "plt.plot(fpr_lr, tpr_lr, color=\"#1f77b4\", linestyle=\":\", linewidth=2, label=f\"Logistic Regression (AUC = {fresh_auc_lr:.3f})\")\n",
    "plt.plot(fpr_rf, tpr_rf, color=\"#2ca02c\", linestyle=\"-.\", linewidth=2, label=f\"Random Forest (AUC = {fresh_auc_rf:.3f})\")\n",
    "plt.plot(fpr_xgb, tpr_xgb, color=\"#ff7f0e\", linestyle=\"--\", linewidth=2.5, label=f\"XGBoost (AUC = {fresh_auc_xgb:.3f})\")\n",
    "plt.plot(fpr_vote, tpr_vote, color=\"#d62728\", linestyle=\"-\", linewidth=3, label=f\"Soft Voting Ensemble (AUC = {fresh_auc_vote:.3f})\")\n",
    "\n",
    "plt.plot([0, 1], [0, 1], color=\"gray\", linestyle=\"--\", linewidth=1, label=\"Random Guessing (AUC = 0.500)\")\n",
    "\n",
    "plt.title(\"Receiver Operating Characteristic (ROC) - Centralized Baselines\", fontsize=14, fontweight=\"bold\", pad=15)\n",
    "plt.xlabel(\"False Positive Rate\", fontsize=12, fontweight=\"bold\")\n",
    "plt.ylabel(\"True Positive Rate\", fontsize=12, fontweight=\"bold\")\n",
    "plt.legend(loc=\"lower right\", fontsize=10, frameon=True, shadow=True)\n",
    "plt.grid(True, linestyle=\"--\", alpha=0.6)\n",
    "\n",
    "plt.xlim([0.0, 1.0])\n",
    "plt.ylim([0.0, 1.05])\n",
    "\n",
    "save_path_pdf = \"../4_evaluation_and_paper/plots/unified_roc_curve.pdf\"\n",
    "save_path_png = \"../4_evaluation_and_paper/plots/unified_roc_curve.png\"\n",
    "\n",
    "plt.savefig(save_path_pdf, format=\"pdf\", dpi=300, bbox_inches=\"tight\")\n",
    "plt.savefig(save_path_png, format=\"png\", dpi=300, bbox_inches=\"tight\")\n",
    "\n",
    "plt.show()\n"
]

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": new_plot_code
}

# Remove existing plotting block if it's the last cell (we know from previous investigation)
if len(nb["cells"]) > 0 and nb["cells"][-1]["cell_type"] == "code" and "plotting code" in "".join(nb["cells"][-1]["source"]):
    nb["cells"].pop()

# Add the new cell immediately after visuals_idx
nb["cells"].insert(visuals_idx + 1, new_cell)

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
