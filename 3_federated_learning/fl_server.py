"""
fl_server.py
============
Federated Learning Server — FedProx strategy with:
  • Macro-averaging across NY and TX hospital clients
  • ModelSavingFedProx: saves aggregated global model after final round
    → output/models/federated_mlp.pt   (used by generate_figures.py)
  • save_history: serialises per-round Loss + AUC to JSON after training
    → output/fl_training_history.json  (Fig 1: FL Convergence Curve)
"""

import json
import os
import sys
import torch
import flwr as fl
from typing import List, Tuple, Dict, Optional, Union
from flwr.common import (
    Metrics, Parameters, FitRes, Scalar,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from collections import OrderedDict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SERVER_DIR      = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR         = os.path.join(_SERVER_DIR, "..")
HISTORY_OUT_PATH = os.path.join(ROOT_DIR, "output", "fl_training_history.json")
MODEL_OUT_PATH   = os.path.join(ROOT_DIR, "output", "models", "federated_mlp.pt")
NUM_ROUNDS       = 50

# model.py lives in the same directory as fl_server.py
if _SERVER_DIR not in sys.path:
    sys.path.insert(0, _SERVER_DIR)
from model import DiabetesMLP  # noqa: E402


# ---------------------------------------------------------------------------
# Macro-averaging (IEEE: treat each hospital as an equal partner)
# ---------------------------------------------------------------------------
def macro_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    IEEE Research Fix: Macro-averaging treats every hospital as an equal 1-to-1 partner
    during evaluation, regardless of patient volume.
    """
    num_clients = len(metrics)  # Will be 2 (NY and TX)

    macro_acc    = sum([m["accuracy"] for _, m in metrics]) / num_clients
    macro_auc    = sum([m["auc"]      for _, m in metrics]) / num_clients
    macro_recall = sum([m["recall"]   for _, m in metrics]) / num_clients
    macro_f1     = sum([m["f1"]       for _, m in metrics]) / num_clients
    macro_loss   = sum([m["loss"]     for _, m in metrics]) / num_clients

    return {
        "accuracy": macro_acc,
        "auc":      macro_auc,
        "recall":   macro_recall,
        "f1":       macro_f1,
        "loss":     macro_loss,
    }


# ---------------------------------------------------------------------------
# Custom Strategy: saves the final global model after round NUM_ROUNDS
# ---------------------------------------------------------------------------
class ModelSavingFedProx(fl.server.strategy.FedProx):
    """
    Extends FedProx to save the aggregated global model weights after the
    final communication round → output/models/federated_mlp.pt.
    This file is consumed by generate_figures.py for Figures 2, 3, and 4.
    """

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        # Standard FedProx aggregation first
        aggregated_params, metrics = super().aggregate_fit(server_round, results, failures)

        # Save state dict only after the final round
        if aggregated_params is not None and server_round == NUM_ROUNDS:
            self._save_final_model(aggregated_params)

        return aggregated_params, metrics

    def _save_final_model(self, parameters: Parameters) -> None:
        """Reconstruct DiabetesMLP from Flower Parameters and save as .pt file."""
        os.makedirs(os.path.dirname(os.path.abspath(MODEL_OUT_PATH)), exist_ok=True)

        # Flower Parameters → list of numpy arrays → OrderedDict
        ndarrays    = parameters_to_ndarrays(parameters)
        model       = DiabetesMLP(input_dim=21)
        params_dict = zip(model.state_dict().keys(), ndarrays)
        state_dict  = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        torch.save(model.state_dict(), MODEL_OUT_PATH)
        print(f"\n[Server] Final global model saved → {os.path.abspath(MODEL_OUT_PATH)}")


# ---------------------------------------------------------------------------
# History persistence (for Figure 1 — FL Convergence Curve)
# ---------------------------------------------------------------------------
def save_history(history) -> None:
    """
    Persist the Flower History object to JSON so generate_figures.py can
    produce the FL Convergence Curve (Figure 1) without re-running training.

    Flower stores per-round distributed evaluation metrics under:
        history.metrics_distributed  →  {"auc": [(round, val), ...], ...}
        history.losses_distributed   →  [(round, loss), ...]
    """
    os.makedirs(os.path.dirname(os.path.abspath(HISTORY_OUT_PATH)), exist_ok=True)

    rounds_loss = [r        for r, _ in history.losses_distributed]
    loss_values = [float(v) for _, v in history.losses_distributed]

    auc_values = []
    if "auc" in history.metrics_distributed:
        auc_values = [float(v) for _, v in history.metrics_distributed["auc"]]
    elif hasattr(history, "metrics_distributed_fit") and "auc" in history.metrics_distributed_fit:
        auc_values = [float(v) for _, v in history.metrics_distributed_fit["auc"]]

    history_dict = {
        "rounds":     rounds_loss,
        "loss":       loss_values,
        "auc":        auc_values,
        "num_rounds": len(rounds_loss),
    }

    with open(HISTORY_OUT_PATH, "w") as f:
        json.dump(history_dict, f, indent=2)

    print(f"[Server] Training history saved → {os.path.abspath(HISTORY_OUT_PATH)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Starting Federated Learning Server (FedProx + Model Saving)...")

    strategy = ModelSavingFedProx(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=macro_average,
        proximal_mu=1.0,           # Prevents client drift (the Yo-Yo effect)
    )

    # start_server returns a History object after all rounds complete
    history = fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

    # Persist per-round metrics for Figure 1 (FL Convergence Curve)
    save_history(history)


if __name__ == "__main__":
    main()
