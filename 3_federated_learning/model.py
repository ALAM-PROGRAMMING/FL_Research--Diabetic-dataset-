import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score
from warnings import filterwarnings

filterwarnings("ignore")

# Define our robust Multi-Layer Perceptron (MLP) for binary classification
class DiabetesMLP(nn.Module):
    def __init__(self, input_dim=9):
        super(DiabetesMLP, self).__init__()
        
        # Widened to 64 neurons to capture complex tabular interactions
        self.layer1 = nn.Linear(input_dim, 64)
        self.ln1 = nn.LayerNorm(64) # FL MAGIC BULLET: LayerNorm instead of BatchNorm
        self.relu1 = nn.LeakyReLU(0.01) # Prevents dying neurons
        self.dropout1 = nn.Dropout(0.1) # Reduced dropout for tabular data
        
        self.layer2 = nn.Linear(64, 32)
        self.ln2 = nn.LayerNorm(32)
        self.relu2 = nn.LeakyReLU(0.01)
        self.dropout2 = nn.Dropout(0.1)
        
        self.output = nn.Linear(32, 1)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.ln1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.layer2(x)
        x = self.ln2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.output(x)
        return x

def train(model, train_loader, epochs, device, pos_weight=None, global_params=None, proximal_mu=0.0):
    """
    Train the network on the training set.
    pos_weight is CRITICAL for our highly imbalanced medical dataset.
    """
    # Use BCEWithLogitsLoss with pos_weight to handle class imbalance natively in PyTorch
    if pos_weight is not None:
        pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    else:
        criterion = nn.BCEWithLogitsLoss()
        
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device).float()
            # The target needs to be reshaped to match output shape [batch_size, 1]
            target = target.unsqueeze(1)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # IEEE Research Fix: FedProx Proximal Term
            # This penalizes local weights if they drift too far from the global server weights
            if global_params is not None and proximal_mu > 0.0:
                proximal_term = 0.0
                for local_weights, global_weights in zip(model.parameters(), global_params):
                    proximal_term += torch.square((local_weights - global_weights.to(device))).sum()
                loss += (proximal_mu / 2) * proximal_term
                
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * data.size(0)
            
        epoch_loss = epoch_loss / len(train_loader.dataset)
        # Optional: print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

def test(model, test_loader, device):
    """
    Validate the network on the entire test set.
    Returns: loss, accuracy, auc, recall, f1
    """
    criterion = nn.BCEWithLogitsLoss() # standard loss for validation
    model.eval()
    
    total_loss = 0.0
    all_targets = []
    all_outputs = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device).float()
            target = target.unsqueeze(1)
            
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            
            # Since output is logits, apply sigmoid to get probabilities
            probs = torch.sigmoid(output)
            
            all_targets.extend(target.cpu().numpy())
            all_outputs.extend(probs.cpu().numpy())
            
    # Calculate metrics
    avg_loss = total_loss / len(test_loader.dataset)
    
    # Threshold probabilities at 0.5 to get binary predictions
    preds = [1 if p >= 0.5 else 0 for p in all_outputs]
    
    acc = accuracy_score(all_targets, preds)
    try:
        # ROC AUC requires both classes to be present in target
        auc = roc_auc_score(all_targets, all_outputs)
    except ValueError:
        auc = 0.5
        
    recall = recall_score(all_targets, preds, zero_division=0)
    f1 = f1_score(all_targets, preds, zero_division=0)
    
    return avg_loss, acc, auc, recall, f1
