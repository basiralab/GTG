import os
import torch
import numpy as np
from torch_geometric.data import Data
import torch.nn.functional as F
from models.STP_GSR import STPGSR, revert_dual, create_dual_graph_feature_matrix
from utils import load_data
import time
import psutil
import gc
from datetime import datetime
from sklearn.model_selection import KFold
import tempfile
from tqdm import tqdm

# Configuration class for STP-GSR
class STPGSRConfig:
    def __init__(self, n_source_nodes, n_target_nodes):
        self.dataset = DatasetConfig(n_source_nodes, n_target_nodes)
        self.model = ModelConfig()
        self.experiment = ExperimentConfig()

class DatasetConfig:
    def __init__(self, n_source_nodes, n_target_nodes):
        self.n_source_nodes = n_source_nodes
        self.n_target_nodes = n_target_nodes
        self.node_feat_init = 'adj'
        self.node_feat_dim = n_source_nodes

class ModelConfig:
    def __init__(self):
        self.name = 'stp_gsr'
        self.target_edge_initializer = TargetEdgeInitializerConfig()
        self.dual_learner = DualLearnerConfig()

class TargetEdgeInitializerConfig:
    def __init__(self):
        self.num_heads = 1
        self.edge_dim = 1
        self.dropout = 0.2
        self.beta = False

class DualLearnerConfig:
    def __init__(self):
        self.in_dim = 1
        self.out_dim = 1
        self.num_heads = 1
        self.dropout = 0.2
        self.beta = False

class ExperimentConfig:
    def __init__(self):
        self.n_epochs = 50
        self.batch_size = 8
        self.lr = 0.001
        self.log_val_loss = True

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def log_metrics(log_file, metrics):
    """Log metrics to file with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, 'a') as f:
        f.write(f"\n[{timestamp}] {metrics}\n")

def prepare_graph_data(adj_matrix):
    """Convert adjacency matrix to PyTorch Geometric Data object"""
    # Convert numpy array to tensor if needed
    if isinstance(adj_matrix, np.ndarray):
        adj_matrix = torch.from_numpy(adj_matrix).float()
    
    n = adj_matrix.shape[0]
    edge_index = torch.nonzero(adj_matrix).t()
    x = adj_matrix  # Use adjacency matrix as node features
    
    return Data(
        x=x, 
        pos_edge_index=edge_index,
        num_nodes=n, 
        adj=adj_matrix
    )

def adapt_to_stp_gsr_format(data_list):
    """Convert data to STP-GSR format"""
    adapted_data = []
    for data in data_list:
        # Create edge attributes from adjacency matrix
        edge_attr = data.adj[data.edge_index[0], data.edge_index[1]].unsqueeze(1)
        
        # Create new Data object with required attributes for STP-GSR
        adapted_data.append(Data(
            x=data.adj,  # Use adjacency matrix as node features
            pos_edge_index=data.edge_index,  # Use existing edge indices
            edge_attr=edge_attr,  # Create edge attributes from adj matrix
            adj=data.adj  # Keep original adjacency matrix
        ))
    return adapted_data

def create_stp_gsr_data_dict(pyg_data, adj_matrix):
    """Create data dictionary format expected by STP-GSR"""
    return {
        'pyg': pyg_data,
        'mat': adj_matrix
    }

def train_stp_gsr(config, source_data, target_data, device):
    """Train STP-GSR model"""
    # Initialize model
    model = STPGSR(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.experiment.lr)
    criterion = torch.nn.L1Loss()
    
    train_losses = []
    val_losses = []
    
    print(f"Training STP-GSR model for {config.experiment.n_epochs} epochs...")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        model.train()
        step_counter = 0
        
        for epoch in range(config.experiment.n_epochs):
            batch_counter = 0
            epoch_loss = 0.0
            
            # Shuffle training data
            random_idx = torch.randperm(len(source_data))
            source_train = [source_data[i] for i in random_idx]
            target_train = [target_data[i] for i in random_idx]
            
            # Train on each sample
            for source, target in tqdm(zip(source_train, target_train), 
                                     total=len(source_train), 
                                     desc=f"Epoch {epoch+1}/{config.experiment.n_epochs}"):
                
                source_g = source['pyg'].to(device)
                source_m = source['mat'].to(device)
                target_m = target['mat'].to(device)
                
                # Forward pass
                model_pred, model_target = model(source_g, target_m)
                
                # Calculate loss
                loss = criterion(model_pred, model_target)
                loss.backward()
                
                epoch_loss += loss.item()
                batch_counter += 1
                
                # Gradient accumulation and optimization
                if batch_counter % config.experiment.batch_size == 0 or batch_counter == len(source_train):
                    optimizer.step()
                    optimizer.zero_grad()
                    step_counter += 1
                    
                    # Memory cleanup
                    torch.cuda.empty_cache()
                    gc.collect()
            
            # Calculate average epoch loss
            epoch_loss = epoch_loss / len(source_train)
            train_losses.append(epoch_loss)
            
            print(f'Epoch {epoch+1}/{config.experiment.n_epochs}, Train Loss: {epoch_loss:.4f}')
            
            # Validation (every 10 epochs to save time)
            if config.experiment.log_val_loss and (epoch + 1) % 10 == 0:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for source, target in zip(source_data, target_data):
                        source_g = source['pyg'].to(device)
                        target_m = target['mat'].to(device)
                        
                        model_pred, model_target = model(source_g, target_m)
                        val_loss += criterion(model_pred, model_target).item()
                
                val_loss = val_loss / len(source_data)
                val_losses.append(val_loss)
                print(f'Epoch {epoch+1}/{config.experiment.n_epochs}, Val Loss: {val_loss:.4f}')
                model.train()
    
    return model, train_losses, val_losses

def evaluate_stp_gsr(model, source_data, target_data, device, n_target_nodes):
    """Evaluate STP-GSR model"""
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for source, target in zip(source_data, target_data):
            source_g = source['pyg'].to(device)
            target_m = target['mat'].to(device)
            
            # Get prediction
            model_pred, _ = model(source_g, target_m)
            
            # Convert dual graph prediction back to adjacency matrix
            pred_adj = revert_dual(model_pred, n_target_nodes)
            
            predictions.append(pred_adj.cpu().numpy())
            targets.append(target_m.cpu().numpy())
    
    model.train()
    return predictions, targets

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    supervised = False
    
    # Create output directory
    if supervised:
        output_dir = "results/outputs/supervised_stp_gsr_cv_asd_lh"
    else:
        output_dir = "results/outputs/stp_gsr_cv_asd_lh"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create log file
    log_file = os.path.join(output_dir, 'training_metrics.txt')
    with open(log_file, 'w') as f:
        f.write("STP-GSR Training Metrics Log\n")
        f.write("===========================\n")
        f.write(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}\n")
    
    # Dataset parameters
    n_source_nodes = 35
    n_target_nodes = 35
    
    # Cross-validation setup
    cv_dir = "dataset/5F_CV_asd_lh_dataset"
    fold_indices = [1, 2, 3, 4]
    test_fold = 5
    
    for val_fold in fold_indices:
        train_folds = [f for f in fold_indices if f != val_fold]
        print(f"\n=== Train on folds {train_folds}, Validate on fold {val_fold}, Test on fold {test_fold} ===")
        
        # Log fold start
        log_metrics(log_file, f"Starting fold: Train={train_folds}, Val={val_fold}, Test={test_fold}")
        fold_start_time = time.time()
        initial_memory = get_memory_usage()
        log_metrics(log_file, f"Initial memory usage: {initial_memory:.2f} MB")
        
        # Gather training data
        train_source = []
        train_target = []
        for f in train_folds:
            fold_dir = os.path.join(cv_dir, f"fold_{f}")
            train_src_path = os.path.join(fold_dir, f"X_train_{f}.csv")
            if supervised:
                train_trg_path = os.path.join(fold_dir, f"Y_train_{f}.csv")
            else:
                train_trg_path = train_src_path
            src, trg = load_data(train_src_path, train_trg_path, 
                               node_size=n_source_nodes, 
                               target_node_size=n_target_nodes,
                               feature_strategy="adj")
            train_source.extend(src)
            train_target.extend(trg)
        
        # Adapt data format for STP-GSR
        train_source = adapt_to_stp_gsr_format(train_source)
        train_target = adapt_to_stp_gsr_format(train_target)
        
        # Convert to STP-GSR data format
        train_source_dict = [create_stp_gsr_data_dict(src, src.adj) for src in train_source]
        train_target_dict = [create_stp_gsr_data_dict(trg, trg.adj) for trg in train_target]
        
        # Create configuration
        config = STPGSRConfig(n_source_nodes, n_target_nodes)
        log_metrics(log_file, f"Configuration: {n_source_nodes} -> {n_target_nodes} nodes")
        
        # Train model
        print("Training STP-GSR...")
        model, train_losses, val_losses = train_stp_gsr(config, train_source_dict, train_target_dict, device)
        
        # Save model
        model_path = os.path.join(output_dir, f"stp_gsr_train{''.join(map(str,train_folds))}_val{val_fold}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved as {model_path}")
        
        # Save training metrics
        np.save(os.path.join(output_dir, f"train_losses_fold{val_fold}.npy"), np.array(train_losses))
        if val_losses:
            np.save(os.path.join(output_dir, f"val_losses_fold{val_fold}.npy"), np.array(val_losses))
        
        # Load validation data for evaluation
        val_fold_dir = os.path.join(cv_dir, f"fold_{val_fold}")
        val_src_path = os.path.join(val_fold_dir, f"X_train_{val_fold}.csv")
        if supervised:
            val_trg_path = os.path.join(val_fold_dir, f"Y_train_{val_fold}.csv")
        else:
            val_trg_path = val_src_path
        val_source, val_target = load_data(val_src_path, val_trg_path, 
                                         node_size=n_source_nodes, 
                                         target_node_size=n_target_nodes,
                                         feature_strategy="adj")
        
        # Adapt validation data
        val_source = adapt_to_stp_gsr_format(val_source)
        val_target = adapt_to_stp_gsr_format(val_target)
        val_source_dict = [create_stp_gsr_data_dict(src, src.adj) for src in val_source]
        val_target_dict = [create_stp_gsr_data_dict(trg, trg.adj) for trg in val_target]
        
        # Evaluate on validation set
        print("Evaluating on validation set...")
        val_predictions, val_targets = evaluate_stp_gsr(model, val_source_dict, val_target_dict, device, n_target_nodes)
        
        # Save validation results
        np.save(os.path.join(output_dir, f"val_predictions_fold{val_fold}.npy"), np.array(val_predictions))
        np.save(os.path.join(output_dir, f"val_targets_fold{val_fold}.npy"), np.array(val_targets))
        
        # Calculate and log fold summary
        fold_time = time.time() - fold_start_time
        final_memory = get_memory_usage()
        final_train_loss = train_losses[-1] if train_losses else 0
        final_val_loss = val_losses[-1] if val_losses else 0
        
        log_metrics(log_file, 
            f"Fold Summary:\n"
            f"Total time: {fold_time:.2f}s\n"
            f"Final memory usage: {final_memory:.2f}MB\n"
            f"Final train loss: {final_train_loss:.4f}\n"
            f"Final val loss: {final_val_loss:.4f}"
        )
        
        # Clear memory
        del model, train_source_dict, train_target_dict, val_source_dict, val_target_dict
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Log final summary
    log_metrics(log_file, 
        f"Training Complete\n"
        f"================\n"
        f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Final memory usage: {get_memory_usage():.2f}MB"
    )
    
    print(f"\nTraining metrics logged to {log_file}")
    print(f"Models and results saved to {output_dir}")

if __name__ == "__main__":
    main() 