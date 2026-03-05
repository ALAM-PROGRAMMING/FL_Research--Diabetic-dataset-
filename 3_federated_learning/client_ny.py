import os
import torch
import flwr as fl
import pandas as pd
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from model import DiabetesMLP, train, test

# 1. IEEE Standard: Strict Environmental Control
# Force deterministic behavior where possible
torch.manual_seed(42)

def load_data():
    """Load and preprocess the US dataset strictly locally."""
    print("Loading US Dataset (250k+ rows)...")
    
    # Strictly bind this client to the NY dataset only. True decentralization.
    df = pd.read_csv("../data/processed/Hospital_NY.csv")
    
    X = df.drop(columns=["Diabetes_binary"]).values
    y = df["Diabetes_binary"].values
    
    # IEEE Standard: Zero Data Leakage. Split BEFORE scaling.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Calculate pos_weight for handling the medical class imbalance
    # (Number of Healthy) / (Number of Diabetic)
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"US Dataset Positional Weight Calculated: {pos_weight:.3f}")
    
    # Convert to PyTorch tensors
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                                  torch.tensor(y_train, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), 
                                 torch.tensor(y_test, dtype=torch.float32))
    
    # DataLoader batch sizes - highly impactful for Deep Learning
    # Using 256 to force frequent weight updates
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256)
    
    return train_loader, test_loader, pos_weight

class USHospitalClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, pos_weight, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.pos_weight = pos_weight
        self.device = device

    def get_parameters(self, config):
        """Extract PyTorch weights to send to the Server."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Apply aggregated weights received from the Server."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Train the model on the local US dataset."""
        print("  Client US: Received weights from server. Training with FedProx...")
        self.set_parameters(parameters)
        
        # IEEE Standard FedProx: The parameters received ARE the global model.
        # We must freeze them as tensors to calculate the proximal loss locally.
        global_params = [torch.tensor(p).to(self.device).float() for p in parameters]
        
        # Train for 5 local epochs (Safe now because FedProx proximal_mu=1.0 prevents drift)
        train(self.model, self.train_loader, epochs=5, device=self.device, 
              pos_weight=self.pos_weight, global_params=global_params, proximal_mu=1.0)
        
        # Return updated weights and the number of training examples
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        """Evaluate the global model on the local US test set."""
        print("  Client US: Evaluating global model on local test set...")
        self.set_parameters(parameters)
        
        loss, accuracy, auc, recall, f1 = test(self.model, self.test_loader, self.device)
        
        print(f"    US Eval -> Loss: {loss:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}")
        
        # Flwr expects loss, num_examples, and a dictionary of custom metrics
        return loss, len(self.test_loader.dataset), {
            "accuracy": accuracy, 
            "auc": auc, 
            "recall": recall, 
            "f1": f1,
            "loss": loss
        }

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"US Client running on device: {device}")
    
    # 1. Load US Data safely
    train_loader, test_loader, pos_weight = load_data()
    
    # 2. Instantiate the global blueprint
    model = DiabetesMLP(input_dim=21).to(device)
    
    # 3. Start the Flower Client connected to localhost:8080
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=USHospitalClient(model, train_loader, test_loader, pos_weight, device).to_client(),
    )

if __name__ == "__main__":
    main()
