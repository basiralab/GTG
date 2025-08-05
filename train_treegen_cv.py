import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.TreeGen import PairwiseDecoder, TreeEncoder, CrossTreeAggregator
from utils import load_data
from train_treegen import GraphPairDataset, GraphGenModel, train_epoch, validate
import numpy as np
import matplotlib.pyplot as plt
import os
import itertools
import time
import psutil
import gc
from datetime import datetime

# Hyperparameters
node_size = 35
feature_strategy = "adj"
hidden_dim = 32
num_layers = 2
out_dim = 16
num_roots = 15
k_hop = 1
batch_size = 8
num_epochs = 50
learning_rate = 1e-3
alpha = 0
beta = 5.0
seed = 42
output_dir = "results/outputs/treegen_cv_asd_lh_" + str(num_roots) + "_weight"
cv_dir = "dataset/5F_CV_asd_lh_dataset"

os.makedirs(output_dir, exist_ok=True)

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
    f.write(f"Number of roots: {num_roots}\n")
    f.write(f"k-hop: {k_hop}\n")

fold_indices = [1, 2, 3, 4]
test_fold = 5

all_train_losses = []
all_val_losses = []
all_test_losses = []

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

    train_dataset = GraphPairDataset(train_src, train_trg, m=num_roots, k=k_hop)
    val_dataset = GraphPairDataset(val_src, val_trg, m=num_roots, k=k_hop)
    test_dataset = GraphPairDataset(test_src, test_trg, m=num_roots, k=k_hop)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

    # Model
    tree_encoder = TreeEncoder(
        in_channels=node_size,
        hidden_channels=hidden_dim,
        num_layers=num_layers,
        out_channels=out_dim
    )
    aggregator = CrossTreeAggregator(
        in_node_dim=node_size,
        in_tree_dim=out_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_layers=num_layers
    )
    decoder = PairwiseDecoder(
        node_dim=out_dim,
        hidden_dim=hidden_dim
    )
    model = GraphGenModel(tree_encoder, aggregator, decoder)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    struct_loss_fn = nn.BCEWithLogitsLoss()
    weight_loss_fn = nn.L1Loss()

    train_struct_losses, train_weight_losses = [], []
    val_struct_losses, val_weight_losses = [], []
    test_struct_losses, test_weight_losses = [], []

    best_val_loss = float('inf')
    best_model_state = None

    # Track epoch times and memory usage
    epoch_times = []
    epoch_memory_usage = []

    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()
        epoch_start_memory = get_memory_usage()
        
        # Training
        train_epoch_losses = train_epoch(
            model, train_loader, device='cpu',
            struct_loss_fn=struct_loss_fn,
            weight_loss_fn=weight_loss_fn,
            alpha=alpha,
            beta=beta,
            optimizer=optimizer
        )
        
        # Validation
        validate_losses = validate(
            model, val_loader, device='cpu',
            struct_loss_fn=struct_loss_fn,
            weight_loss_fn=weight_loss_fn,
            alpha=alpha,
            beta=beta
        )
        
        # Record losses
        train_struct_losses.append(train_epoch_losses['struct_loss'])
        train_weight_losses.append(train_epoch_losses['weight_loss'])
        val_struct_losses.append(validate_losses['struct_loss'])
        val_weight_losses.append(validate_losses['weight_loss'])

        # Save best model by validation struct+weight loss
        val_total_loss = validate_losses['struct_loss'] + validate_losses['weight_loss']
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
            f"Train Struct: {train_struct_losses[-1]:.4f}, "
            f"Train Weight: {train_weight_losses[-1]:.4f}, "
            f"Val Struct: {val_struct_losses[-1]:.4f}, "
            f"Val Weight: {val_weight_losses[-1]:.4f}"
        )

        print(f"Train folds {train_folds} | Val fold {val_fold} | Epoch {epoch}: "
              f"Train Struct={train_struct_losses[-1]:.4f}, Train Weight={train_weight_losses[-1]:.4f} | "
              f"Val Struct={val_struct_losses[-1]:.4f}, Val Weight={val_weight_losses[-1]:.4f}")

    # Test after training
    test_start_time = time.time()
    test_losses = validate(
        model, test_loader, device='cpu',
        struct_loss_fn=struct_loss_fn,
        weight_loss_fn=weight_loss_fn,
        alpha=alpha,
        beta=beta
    )
    test_time = time.time() - test_start_time
    test_struct_losses.append(test_losses['struct_loss'])
    test_weight_losses.append(test_losses['weight_loss'])

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
    all_train_losses.append((train_struct_losses, train_weight_losses))
    all_val_losses.append((val_struct_losses, val_weight_losses))
    all_test_losses.append((test_struct_losses[0], test_weight_losses[0]))

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
        f"Test Struct Loss: {test_losses['struct_loss']:.4f}\n"
        f"Test Weight Loss: {test_losses['weight_loss']:.4f}"
    )

    # Plot loss curves for this split
    plt.figure()
    plt.plot(train_struct_losses, label='Train Struct Loss')
    plt.plot(val_struct_losses, label='Val Struct Loss')
    plt.plot(train_weight_losses, label='Train Weight Loss')
    plt.plot(val_weight_losses, label='Val Weight Loss')
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
test_structs = [x[0] for x in all_test_losses]
test_weights = [x[1] for x in all_test_losses]
x_labels = [f"Train{''.join(str(f) for f in fold_indices if f!=val)}_Val{val}" for val in fold_indices]
plt.bar(np.arange(4)-0.15, test_structs, width=0.3, label='Test Struct Loss')
plt.bar(np.arange(4)+0.15, test_weights, width=0.3, label='Test Weight Loss')
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