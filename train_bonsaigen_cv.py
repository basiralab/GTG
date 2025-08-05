import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import psutil
import gc
from datetime import datetime
import pandas as pd

from utils import load_data, GraphTreeProducer, WLConfig
from models.BonsaiGen import ExemplarGNN2AdjModel, ExemplarGraphDataset

# Hyperparameters
node_size = 35
feature_strategy = "adj"
wl_iterations = 2
k = 10
frac_to_sample = 1.0
hidden_dim = 64
num_epochs = 50
batch_size = 1
learning_rate = 1e-3
alpha = 10.0
beta = 5.0
seed = 42
output_dir = "results/outputs/bonsai_cv_asd_lh"
cv_dir = "dataset/5F_CV_asd_lh_dataset"
model_type = "mlp"

os.makedirs(output_dir, exist_ok=True)

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def log_metrics(log_file, metrics):
    """Log metrics to file with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, 'a') as f:
        f.write(f"\n[{timestamp}] {metrics}\n")

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

set_seed(seed)

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
    f.write(f"Number of exemplars (k): {k}\n")
    f.write(f"WL iterations: {wl_iterations}\n")

fold_indices = [1, 2, 3, 4]
test_fold = 5

all_train_losses = []
all_val_losses = []
all_test_losses = []
all_training_times = []
all_memory_usages = []

def custom_collate(batch):
    return batch

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

    # Log data loading completion
    log_metrics(log_file, f"Data loading completed. Memory usage: {get_memory_usage():.2f} MB")

    wl_config = WLConfig(add_self_loops=True, normalize_adj=True, iterations=wl_iterations, symmetric=True)

    train_dataset = ExemplarGraphDataset(train_src, train_trg, wl_config, k=k, frac_to_sample=frac_to_sample)
    val_dataset = ExemplarGraphDataset(val_src, val_trg, wl_config, k=k, frac_to_sample=frac_to_sample)
    test_dataset = ExemplarGraphDataset(test_src, test_trg, wl_config, k=k, frac_to_sample=frac_to_sample)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

    # Get input dimension from first sample
    in_dim = train_dataset[0][0].shape[1]
    num_exemplars = train_dataset[0][0].shape[0]

    model = ExemplarGNN2AdjModel(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_exemplars=num_exemplars,
        out_nodes=node_size,
        model_type=model_type,
        heads=2,
        pairwise_decoder=True
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_mse = nn.MSELoss()

    train_bce_losses, train_mse_losses = [], []
    val_bce_losses, val_mse_losses = [], []
    test_bce_losses, test_mse_losses = [], []
    epoch_times = []
    epoch_memory_usage = []

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()
        epoch_start_memory = get_memory_usage()
        
        model.train()
        train_bce_loss = 0
        train_mse_loss = 0
        for batch in train_loader:
            for exemplar_wl, edge_index, target_adj in batch:
                exemplar_wl = exemplar_wl.squeeze(0)
                edge_index = edge_index.squeeze(0)
                target_adj = target_adj.squeeze(0)
                pred_adj = model(exemplar_wl, edge_index)
                loss_bce = criterion_bce(pred_adj, target_adj)
                loss_mse = criterion_mse(torch.sigmoid(pred_adj), target_adj)
                loss = alpha * loss_bce + beta * loss_mse
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_bce_loss += loss_bce.item()
                train_mse_loss += loss_mse.item()
        n_train = len(train_loader.dataset)
        train_bce_losses.append(train_bce_loss / n_train)
        train_mse_losses.append(train_mse_loss / n_train)

        # Validation
        model.eval()
        val_bce_loss = 0
        val_mse_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                for exemplar_wl, edge_index, target_adj in batch:
                    exemplar_wl = exemplar_wl.squeeze(0)
                    edge_index = edge_index.squeeze(0)
                    target_adj = target_adj.squeeze(0)
                    pred_adj = model(exemplar_wl, edge_index)
                    loss_bce = criterion_bce(pred_adj, target_adj)
                    loss_mse = criterion_mse(torch.sigmoid(pred_adj), target_adj)
                    val_bce_loss += loss_bce.item()
                    val_mse_loss += loss_mse.item()
        n_val = len(val_loader.dataset)
        val_bce_losses.append(val_bce_loss / n_val)
        val_mse_losses.append(val_mse_loss / n_val)

        # Save best model by validation total loss
        val_total_loss = val_bce_losses[-1] + val_mse_losses[-1]
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
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
            f"Train BCE: {train_bce_losses[-1]:.4f}, "
            f"Train MSE: {train_mse_losses[-1]:.4f}, "
            f"Val BCE: {val_bce_losses[-1]:.4f}, "
            f"Val MSE: {val_mse_losses[-1]:.4f}"
        )

        print(f"Train folds {train_folds} | Val fold {val_fold} | Epoch {epoch}: "
              f"Train BCE={train_bce_losses[-1]:.4f}, Train MSE={train_mse_losses[-1]:.4f} | "
              f"Val BCE={val_bce_losses[-1]:.4f}, Val MSE={val_mse_losses[-1]:.4f} | "
              f"Time={epoch_time:.2f}s")

    # Test after training
    test_start_time = time.time()
    model.eval()
    test_bce_loss = 0
    test_mse_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            for exemplar_wl, edge_index, target_adj in batch:
                exemplar_wl = exemplar_wl.squeeze(0)
                edge_index = edge_index.squeeze(0)
                target_adj = target_adj.squeeze(0)
                pred_adj = model(exemplar_wl, edge_index)
                loss_bce = criterion_bce(pred_adj, target_adj)
                loss_mse = criterion_mse(torch.sigmoid(pred_adj), target_adj)
                test_bce_loss += loss_bce.item()
                test_mse_loss += loss_mse.item()
    test_time = time.time() - test_start_time
    n_test = len(test_loader.dataset)
    test_bce_losses.append(test_bce_loss / n_test)
    test_mse_losses.append(test_mse_loss / n_test)

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
    all_train_losses.append((train_bce_losses, train_mse_losses))
    all_val_losses.append((val_bce_losses, val_mse_losses))
    all_test_losses.append((test_bce_losses[-1], test_mse_losses[-1]))

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
        f"Test BCE Loss: {test_bce_losses[-1]:.4f}\n"
        f"Test MSE Loss: {test_mse_losses[-1]:.4f}"
    )

    # Plot loss curves for this split
    plt.figure()
    plt.plot(train_bce_losses, label='Train BCE Loss')
    plt.plot(val_bce_losses, label='Val BCE Loss')
    plt.plot(train_mse_losses, label='Train MSE Loss')
    plt.plot(val_mse_losses, label='Val MSE Loss')
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
test_bces = [x[0] for x in all_test_losses]
test_mses = [x[1] for x in all_test_losses]
x_labels = [f"Train{''.join(str(f) for f in fold_indices if f!=val)}_Val{val}" for val in fold_indices]
plt.bar(np.arange(4)-0.15, test_bces, width=0.3, label='Test BCE Loss')
plt.bar(np.arange(4)+0.15, test_mses, width=0.3, label='Test MSE Loss')
plt.xticks(np.arange(4), x_labels, rotation=20)
plt.ylabel('Loss')
plt.title('Final Test Losses for Each Cross-Validation Split')
plt.legend()
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