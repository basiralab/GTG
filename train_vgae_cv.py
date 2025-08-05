import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from utils import load_data
import numpy as np
import matplotlib.pyplot as plt
import os
from torch_geometric.nn import GCNConv
import time
import psutil
import gc
from datetime import datetime
from models.VGAE import VGAE, VGAEDataset

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def log_metrics(log_file, metrics):
    """Log metrics to file with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, 'a') as f:
        f.write(f"\n[{timestamp}] {metrics}\n")

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in loader:
        # Process each item in the batch
        for x, edge_index, adj_matrix in batch:
            x, edge_index, adj_matrix = x.to(device), edge_index.to(device), adj_matrix.to(device)
            
            optimizer.zero_grad()
            adj_recon, mu, logstd = model(x, edge_index)
            
            # Reconstruction loss
            recon_loss = F.binary_cross_entropy(adj_recon, adj_matrix)
            
            # KL divergence loss
            kl_loss = -0.5 * torch.mean(torch.sum(1 + logstd - mu.pow(2) - logstd.exp(), dim=1))
            
            # Total loss
            loss = recon_loss + kl_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
    
    return total_loss / len(loader)

def validate(model, loader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in loader:
            # Process each item in the batch
            for x, edge_index, adj_matrix in batch:
                x, edge_index, adj_matrix = x.to(device), edge_index.to(device), adj_matrix.to(device)
                
                adj_recon, mu, logstd = model(x, edge_index)
                
                # Reconstruction loss
                recon_loss = F.binary_cross_entropy(adj_recon, adj_matrix)
                
                # KL divergence loss
                kl_loss = -0.5 * torch.mean(torch.sum(1 + logstd - mu.pow(2) - logstd.exp(), dim=1))
                
                # Total loss
                loss = recon_loss + kl_loss
                total_loss += loss.item()
    
    return total_loss / len(loader)

def custom_collate(batch):
    return batch

# Hyperparameters
node_size = 35
feature_strategy = "adj"
hidden_dim = 32
out_dim = 16
batch_size = 8
num_epochs = 50
learning_rate = 1e-3
seed = 42
output_dir = "results/outputs/vgae_cv_asd_lh"
cv_dir = "dataset/5F_CV_asd_lh_dataset"

os.makedirs(output_dir, exist_ok=True)

# Create log file
log_file = os.path.join(output_dir, 'training_metrics.txt')
with open(log_file, 'w') as f:
    f.write("Training Metrics Log\n")
    f.write("===================\n")
    f.write(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}\n")
    f.write(f"Batch size: {batch_size}\n")
    f.write(f"Number of epochs: {num_epochs}\n")
    f.write(f"Learning rate: {learning_rate}\n")
    f.write(f"Hidden dimension: {hidden_dim}\n")
    f.write(f"Output dimension: {out_dim}\n")

set_seed(seed)

fold_indices = [1, 2, 3, 4]
test_fold = 5

all_train_losses = []
all_val_losses = []
all_test_losses = []

for val_fold in fold_indices:
    train_folds = [f for f in fold_indices if f != val_fold]
    print(f"\n=== Train on folds {train_folds}, Validate on fold {val_fold}, Test on fold {test_fold} ===")
    
    # Log fold start
    log_metrics(log_file, f"Starting fold: Train={train_folds}, Val={val_fold}, Test={test_fold}")
    fold_start_time = time.time()
    initial_memory = get_memory_usage()
    log_metrics(log_file, f"Initial memory usage: {initial_memory:.2f} MB")

    # Gather training data
    train_src, train_trg = [], []
    for f in train_folds:
        fold_dir = os.path.join(cv_dir, f"fold_{f}")
        train_src_path = os.path.join(fold_dir, f"X_train_{f}.csv")
        train_trg_path = os.path.join(fold_dir, f"X_train_{f}.csv")
        src, trg = load_data(train_src_path, train_trg_path, node_size=node_size, feature_strategy=feature_strategy)
        train_src.extend(src)
        train_trg.extend(trg)

    # Log data loading completion
    log_metrics(log_file, f"Data loading completed. Memory usage: {get_memory_usage():.2f} MB")

    # Validation data
    val_dir = os.path.join(cv_dir, f"fold_{val_fold}")
    val_src_path = os.path.join(val_dir, f"X_train_{val_fold}.csv")
    val_trg_path = os.path.join(val_dir, f"X_train_{val_fold}.csv")
    val_src, val_trg = load_data(val_src_path, val_trg_path, node_size=node_size, feature_strategy=feature_strategy)

    # Test data (always fold 5)
    test_dir = os.path.join(cv_dir, f"fold_{test_fold}")
    test_src_path = os.path.join(test_dir, f"X_train_{test_fold}.csv")
    test_trg_path = os.path.join(test_dir, f"X_train_{test_fold}.csv")
    test_src, test_trg = load_data(test_src_path, test_trg_path, node_size=node_size, feature_strategy=feature_strategy)

    # Create datasets
    train_dataset = VGAEDataset(train_src, train_trg)
    val_dataset = VGAEDataset(val_src, val_trg)
    test_dataset = VGAEDataset(test_src, test_trg)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VGAE(node_size, hidden_dim, out_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None

    # Track epoch times and memory usage
    epoch_times = []
    epoch_memory_usage = []

    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()
        epoch_start_memory = get_memory_usage()
        
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

        # Calculate epoch metrics
        epoch_time = time.time() - epoch_start_time
        epoch_memory = get_memory_usage() - epoch_start_memory
        epoch_times.append(epoch_time)
        epoch_memory_usage.append(epoch_memory)

        # Log epoch metrics
        log_metrics(log_file, 
            f"Epoch {epoch}/{num_epochs} - "
            f"Time: {epoch_time:.2f}s, "
            f"Memory delta: {epoch_memory:.2f}MB, "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}"
        )

        print(f"Train folds {train_folds} | Val fold {val_fold} | Epoch {epoch}: "
              f"Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f}")

    # Test after training
    test_start_time = time.time()
    test_loss = validate(model, test_loader, device)
    test_time = time.time() - test_start_time
    all_test_losses.append(test_loss)

    # Restore best model for this split
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Save best model for this split
    model_save_path = os.path.join(
        output_dir, f"best_model_train{''.join(map(str,train_folds))}_val{val_fold}.pt"
    )
    torch.save(model.state_dict(), model_save_path)
    print(f"Best model saved to {model_save_path}")

    # Save losses for this split
    all_train_losses.append(train_losses)
    all_val_losses.append(val_losses)

    # Calculate and log fold summary
    fold_time = time.time() - fold_start_time
    avg_epoch_time = np.mean(epoch_times)
    max_memory_usage = max(epoch_memory_usage)
    final_memory = get_memory_usage()
    
    log_metrics(log_file, 
        f"Fold Summary:\n"
        f"Total time: {fold_time:.2f}s\n"
        f"Average epoch time: {avg_epoch_time:.2f}s\n"
        f"Maximum memory delta: {max_memory_usage:.2f}MB\n"
        f"Final memory usage: {final_memory:.2f}MB\n"
        f"Test time: {test_time:.2f}s\n"
        f"Test Loss: {test_loss:.4f}"
    )

    # Plot loss curves for this split
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Train folds {train_folds}, Val fold {val_fold}')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'loss_curve_train{"".join(map(str,train_folds))}_val{val_fold}.png'))
    plt.close()

    # Clear memory
    del model, optimizer, train_loader, val_loader, test_loader
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Plot final test losses for all splits
plt.figure()
x_labels = [f"Train{''.join(str(f) for f in fold_indices if f!=val)}_Val{val}" for val in fold_indices]
plt.bar(np.arange(4), all_test_losses)
plt.xticks(np.arange(4), x_labels, rotation=20)
plt.ylabel('Loss')
plt.title('Final Test Losses for Each Cross-Validation Split')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'final_test_losses.png'))
plt.close()

# Log final summary
log_metrics(log_file, 
    f"Training Complete\n"
    f"================\n"
    f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    f"Final memory usage: {get_memory_usage():.2f}MB"
)

print(f"\nAll loss curves and test loss bar plot saved to {output_dir}")
print(f"Training metrics logged to {log_file}")