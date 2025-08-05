import os
import torch
import numpy as np
from torch_geometric.data import Data
import torch.nn.functional as F
from models.IMAN_GraphNet.model import Aligner, Generator, Discriminator
from models.IMAN_GraphNet.losses import *
from models.IMAN_GraphNet.config import *
from utils import load_data, evaluate
import time
import psutil
import gc
from datetime import datetime

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
        pos_edge_index=edge_index,  # Changed from edge_index to pos_edge_index
        num_nodes=n, 
        adj=adj_matrix
    )

def adapt_to_iman_format(data_list):
    """Convert data to IMANGraphNet format"""
    adapted_data = []
    for data in data_list:
        # Create edge attributes from adjacency matrix
        edge_attr = data.adj[data.edge_index[0], data.edge_index[1]].unsqueeze(1)
        
        # Create new Data object with required attributes
        adapted_data.append(Data(
            x=data.adj,  # Use adjacency matrix as node features
            pos_edge_index=data.edge_index,  # Use existing edge indices
            edge_attr=edge_attr,  # Create edge attributes from adj matrix
            adj=data.adj  # Keep original adjacency matrix
        ))
    return adapted_data

def train_iman_graphnet(X_train_source, X_train_target, device):
    """Train IMANGraphNet model"""
    # Initialize models
    aligner = Aligner().to(device)
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    # Initialize optimizers
    Aligner_optimizer = torch.optim.AdamW(aligner.parameters(), lr=0.025, betas=(0.5, 0.999))
    generator_optimizer = torch.optim.AdamW(generator.parameters(), lr=0.025, betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=0.025, betas=(0.5, 0.999))
    
    # Training loop
    for epoch in range(N_EPOCHS):
        aligner.train()
        generator.train()
        discriminator.train()
        
        Al_losses = []
        Ge_losses = []
        losses_discriminator = []
        
        for i in range(len(X_train_source)):
            # Prepare data
            source_graph = X_train_source[i].to(device)
            target_graph = X_train_target[i].to(device)
            
            # Domain alignment
            A_output = aligner(source_graph)
            A_casted = A_output.view(N_SOURCE_NODES, N_SOURCE_NODES)
            
            # Create Data object for generator input
            gen_input = Data(
                x=A_casted,
                pos_edge_index=source_graph.pos_edge_index,  # Use pos_edge_index consistently
                edge_attr=source_graph.edge_attr,
                adj=A_casted
            ).to(device)
            
            # Generate target graph
            G_output = generator(gen_input)
            G_output_reshaped = G_output.view(1, N_TARGET_NODES, N_TARGET_NODES, 1)
            G_output_casted = prepare_graph_data(G_output_reshaped.squeeze().detach().cpu().numpy()).to(device)
            
            # Calculate losses
            target_matrix = target_graph.adj
            
            # Alignment loss
            kl_loss = Alignment_loss(target_matrix, A_output)
            Al_losses.append(kl_loss)
            
            # Generator loss
            Gg_loss = GT_loss(target_matrix, G_output)
            D_real = discriminator(target_graph)
            D_fake = discriminator(G_output_casted)
            G_adversarial = adversarial_loss(D_fake, torch.ones_like(D_fake, requires_grad=False))
            G_loss = G_adversarial + Gg_loss
            Ge_losses.append(G_loss)
            
            # Discriminator loss
            D_real_loss = adversarial_loss(D_real, torch.ones_like(D_real, requires_grad=False))
            D_fake_loss = adversarial_loss(D_fake.detach(), torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2
            losses_discriminator.append(D_loss)
        
        # Update models
        generator_optimizer.zero_grad()
        Ge_losses = torch.mean(torch.stack(Ge_losses))
        Ge_losses.backward(retain_graph=True)
        generator_optimizer.step()
        
        Aligner_optimizer.zero_grad()
        Al_losses = torch.mean(torch.stack(Al_losses))
        Al_losses.backward(retain_graph=True)
        Aligner_optimizer.step()
        
        discriminator_optimizer.zero_grad()
        losses_discriminator = torch.mean(torch.stack(losses_discriminator))
        losses_discriminator.backward(retain_graph=True)
        discriminator_optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'[Epoch: {epoch+1}/{N_EPOCHS}] | [Al loss: {Al_losses:.4f}] | [Ge loss: {Ge_losses:.4f}] | [D loss: {losses_discriminator:.4f}]')
    
    return aligner, generator

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = "results/outputs/supervised_iman_cv_asd_lh"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create log file
    log_file = os.path.join(output_dir, 'training_metrics.txt')
    with open(log_file, 'w') as f:
        f.write("Training Metrics Log\n")
        f.write("===================\n")
        f.write(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}\n")
        f.write(f"Number of epochs: {N_EPOCHS}\n")
    
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
            train_trg_path = os.path.join(fold_dir, f"Y_train_{f}.csv")
            src, trg = load_data(train_src_path, train_trg_path, node_size=35, feature_strategy="adj")
            train_source.extend(src)
            train_target.extend(trg)
        
        # Adapt data format for IMANGraphNet
        train_source = adapt_to_iman_format(train_source)
        train_target = adapt_to_iman_format(train_target)
        
        # Train model
        print("Training IMANGraphNet...")
        aligner, generator = train_iman_graphnet(train_source, train_target, device)
        
        # Save models
        aligner_path = os.path.join(output_dir, f"aligner_train{''.join(map(str,train_folds))}_val{val_fold}.model")
        generator_path = os.path.join(output_dir, f"generator_train{''.join(map(str,train_folds))}_val{val_fold}.model")
        torch.save(aligner.state_dict(), aligner_path)
        torch.save(generator.state_dict(), generator_path)
        
        # Calculate and log fold summary
        fold_time = time.time() - fold_start_time
        final_memory = get_memory_usage()
        
        log_metrics(log_file, 
            f"Fold Summary:\n"
            f"Total time: {fold_time:.2f}s\n"
            f"Final memory usage: {final_memory:.2f}MB"
        )
        
        # Clear memory
        del aligner, generator
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

if __name__ == "__main__":
    main() 