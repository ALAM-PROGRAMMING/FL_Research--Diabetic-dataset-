import flwr as fl
from typing import List, Tuple, Dict, Optional
from flwr.common import Metrics

def macro_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    IEEE Research Fix: Macro-averaging treats every hospital as an equal 1-to-1 partner 
    during evaluation, regardless of patient volume.
    """
    num_clients = len(metrics) # Will be 2 (NY and TX)
    
    # Calculate straight averages, ignoring num_examples entirely
    macro_acc = sum([m["accuracy"] for _, m in metrics]) / num_clients
    macro_auc = sum([m["auc"] for _, m in metrics]) / num_clients
    macro_recall = sum([m["recall"] for _, m in metrics]) / num_clients
    macro_f1 = sum([m["f1"] for _, m in metrics]) / num_clients
    macro_loss = sum([m["loss"] for _, m in metrics]) / num_clients

    # Return the aggregated metrics dictionary for this round
    return {
        "accuracy": macro_acc,
        "auc": macro_auc,
        "recall": macro_recall,
        "f1": macro_f1,
        "loss": macro_loss,
    }

def main():
    print("Starting Federated Learning Server...")
    
    # Define the FedProx strategy
    # FedProx prevents client drift by tethering local updates to the global model.
    strategy = fl.server.strategy.FedProx(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=1.0, # Sample 100% of available clients for evaluation
        min_fit_clients=2, # Wait for both NY and TX before training begins
        min_evaluate_clients=2, # Wait for both to evaluate
        min_available_clients=2, # The server will not start the round until both connect
        evaluate_metrics_aggregation_fn=macro_average, # Our custom logic above
        proximal_mu=1.0, # The mathematical leash that prevents the Yo-Yo effect
    )
    
    # Start the server on port 8080.
    # We will run for 15 rounds of training to allow the complex Tabular MLP to converge.
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=50),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
