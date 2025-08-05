import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from models.TreeGen import PairwiseDecoder, TreeEncoder, CrossTreeAggregator, build_subtrees_from_data
from utils import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import argparse
import random
import os
import networkx as nx
from utils import load_data

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ---------- Graph-to-Graph Model ----------
class GraphGenModel(nn.Module):
    """
    End-to-end model for graph-to-graph generation using subtree encoding, cross-tree aggregation, and pairwise decoding.
    """
    def __init__(self, tree_encoder, aggregator, decoder):
        super().__init__()
        self.tree_encoder = tree_encoder
        self.aggregator   = aggregator
        self.decoder      = decoder

    def forward(self, G, global_feats, subtrees_nodes):
        """
        Args:
            G: networkx.Graph (for extracting subtrees)
            global_feats: [N, F_n] node features
            subtrees_nodes: List[List[int]]
        Returns:
            logits: [N, N] structure logits
            weights: [N, N] edge weights
        """
        # 1) Encode subtrees
        trees = [G.subgraph(nodes).copy() for nodes in subtrees_nodes]
        z_trees = self.tree_encoder(trees, global_feats)  # [m, F_t]
        # 2) Cross aggregation
        h_nodes, _ = self.aggregator(global_feats, z_trees, subtrees_nodes)  # [N, D]
        # 3) Pairwise decode
        logits, weights = self.decoder(h_nodes)
        return logits, weights

# ---------- Training and Validation Loops ----------
def train_epoch(model, data_loader, device, 
                struct_loss_fn, weight_loss_fn, alpha=1.0, beta=1.0, optimizer=None):
    """
    One epoch of training for the graph generation model.
    """
    model.train()
    total_loss = 0.0
    total_struct_loss = 0.0
    total_weight_loss = 0.0
    bar = tqdm(data_loader, desc="Training", leave=False)
    for batch in bar:
        # batch is now a list of dicts, length = batch_size
        batch_loss = 0.0
        batch_struct_loss = 0.0
        batch_weight_loss = 0.0
        for sample in batch:
            G                = sample['G']
            global_feats     = sample['global_feats'].to(device)
            subtrees_nodes   = sample['subtrees_nodes']
            y_struct_target  = sample['target_struct'].to(device)
            y_weight_target  = sample['target_weight'].to(device)

            logits, weights = model(G, global_feats, subtrees_nodes)
            loss_struct = struct_loss_fn(logits, y_struct_target)
            mask = (y_struct_target > 0).float()
            loss_weight = weight_loss_fn(weights * mask, y_weight_target * mask)
            loss = alpha * loss_struct + beta * loss_weight
            
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            batch_loss += loss.item()
            batch_struct_loss += loss_struct.item()
            batch_weight_loss += loss_weight.item()
            
        avg_batch_loss = batch_loss / len(batch)
        avg_batch_struct_loss = batch_struct_loss / len(batch)
        avg_batch_weight_loss = batch_weight_loss / len(batch)
        
        total_loss += avg_batch_loss * len(batch)
        total_struct_loss += avg_batch_struct_loss * len(batch)
        total_weight_loss += avg_batch_weight_loss * len(batch)
        
        bar.set_postfix(loss=avg_batch_loss)
    
    return {
        'total_loss': total_loss / len(data_loader.dataset),
        'struct_loss': total_struct_loss / len(data_loader.dataset),
        'weight_loss': total_weight_loss / len(data_loader.dataset)
    }

def validate(model, data_loader, device, struct_loss_fn, weight_loss_fn, alpha=1.0, beta=1.0):
    """
    Validation loop for the graph generation model.
    """
    model.eval()
    total_loss = 0.0
    total_struct_loss = 0.0
    total_weight_loss = 0.0
    
    with torch.no_grad():
        for batch in data_loader:
            batch_loss = 0.0
            batch_struct_loss = 0.0
            batch_weight_loss = 0.0
            
            for sample in batch:
                G                = sample['G']
                global_feats     = sample['global_feats'].to(device)
                subtrees_nodes   = sample['subtrees_nodes']
                y_struct_target  = sample['target_struct'].to(device)
                y_weight_target  = sample['target_weight'].to(device)

                logits, weights = model(G, global_feats, subtrees_nodes)
                loss_struct = struct_loss_fn(logits, y_struct_target)
                mask = (y_struct_target > 0).float()
                loss_weight = weight_loss_fn(weights * mask, y_weight_target * mask)
                loss = alpha * loss_struct + beta * loss_weight
                
                batch_loss += loss.item()
                batch_struct_loss += loss_struct.item()
                batch_weight_loss += loss_weight.item()
            
            avg_batch_loss = batch_loss / len(batch)
            avg_batch_struct_loss = batch_struct_loss / len(batch)
            avg_batch_weight_loss = batch_weight_loss / len(batch)
            
            total_loss += avg_batch_loss * len(batch)
            total_struct_loss += avg_batch_struct_loss * len(batch)
            total_weight_loss += avg_batch_weight_loss * len(batch)
    
    return {
        'total_loss': total_loss / len(data_loader.dataset),
        'struct_loss': total_struct_loss / len(data_loader.dataset),
        'weight_loss': total_weight_loss / len(data_loader.dataset)
    }

def plot_losses(train_losses, val_losses, save_path='training_losses.png'):
    """
    Plot training and validation losses over epochs.
    """
    epochs = range(1, len(train_losses['total_loss']) + 1)
    plt.figure(figsize=(12, 8))
    
    # Plot total loss
    plt.subplot(3, 1, 1)
    plt.plot(epochs, train_losses['total_loss'], label='Train Total Loss', marker='o')
    plt.plot(epochs, val_losses['total_loss'], label='Val Total Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Total Loss Over Time')
    plt.legend()
    plt.grid(True)
    
    # Plot structure loss
    plt.subplot(3, 1, 2)
    plt.plot(epochs, train_losses['struct_loss'], label='Train Structure Loss', marker='o')
    plt.plot(epochs, val_losses['struct_loss'], label='Val Structure Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Structure Loss Over Time')
    plt.legend()
    plt.grid(True)
    
    # Plot weight loss
    plt.subplot(3, 1, 3)
    plt.plot(epochs, train_losses['weight_loss'], label='Train Weight Loss', marker='o')
    plt.plot(epochs, val_losses['weight_loss'], label='Val Weight Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Weight Loss Over Time')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ---------- Example Dataset ----------
class GraphPairDataset(Dataset):
    def __init__(self, source_data, target_data, m=3, k=2):
        self.source_data = source_data
        self.target_data = target_data
        self.m = m
        self.k = k

    def __getitem__(self, idx):
        # Get PyG Data and adjacency matrix
        src_data = self.source_data[idx]
        trg_data = self.target_data[idx]
        
        # Handle both data formats
        if isinstance(src_data, dict):
            # Synthetic data format from build_data.py
            src_pyg = src_data['pyg']
            target_adj = trg_data['mat']
            src_adj = src_data['mat']
        else:
            # Custom loaded data format from train.py
            src_pyg = src_data
            target_adj = trg_data.adj
            src_adj = src_pyg.adj

        global_feats = src_pyg.x  # [N, F]
        N = global_feats.shape[0]
        
        # Convert adjacency to networkx graph
        adj = src_adj.cpu().numpy() if torch.is_tensor(src_adj) else src_adj
        G = nx.from_numpy_array(adj)
        
        # Create a data dict in the format expected by build_subtrees_from_data
        data_dict = {'mat': adj}
        
        # Extract subtrees (entropy-based roots)
        subtrees = build_subtrees_from_data([data_dict], self.m, self.k, root_strategy='entropy')[0]
        subtrees_nodes = [list(tree.nodes()) for tree in subtrees]
        
        # Target adjacency and weights
        target_weight = target_adj
        target_struct = target_weight.clone()
        
        return {
            "G": G,
            "global_feats": global_feats,
            "subtrees_nodes": subtrees_nodes,
            "target_struct": target_struct,
            "target_weight": target_weight
        }

    def __len__(self):
        return len(self.source_data)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Graph Generation Model')
    
    # Dataset type selection
    parser.add_argument('--dataset_type', type=str, default='file',
                      choices=['built', 'file'],
                      help='Type of dataset to use: "built" for synthetic data or "file" for reading from CSV files')
    
    # Data parameters for built dataset
    parser.add_argument('--n_samples', type=int, default=200,
                      help='Total number of samples (for built dataset)')
    parser.add_argument('--n_source_nodes', type=int, default=35,
                      help='Number of nodes in source graphs')
    parser.add_argument('--n_target_nodes', type=int, default=35,
                      help='Number of nodes in target graphs')
    parser.add_argument('--source_edge_prob', type=float, default=0.2,
                      help='Edge probability for source ER graphs (for built dataset)')
    parser.add_argument('--target_edge_prob', type=float, default=0.3,
                      help='Edge probability for target ER graphs (for built dataset)')
    
    # Data parameters for file dataset
    parser.add_argument('--src_path', type=str, default='dataset/asd_lh_firstlayer_vectors.csv',
                      help='Path to source graph data file (for file dataset)')
    parser.add_argument('--trg_path', type=str, default='dataset/asd_lh_firstlayer_vectors.csv',
                      help='Path to target graph data file (for file dataset)')
    parser.add_argument('--feature_strategy', type=str, default='adj',
                      choices=['one_hot', 'adj', 'degree'],
                      help='Strategy for node features (for file dataset)')
    
    # Model parameters
    parser.add_argument('--node_feat_dim', type=int, default=35,
                      help='Dimension of node features')
    parser.add_argument('--hidden_dim', type=int, default=32,
                      help='Hidden dimension for GNN layers')
    parser.add_argument('--num_layers', type=int, default=2,
                      help='Number of GNN layers')
    parser.add_argument('--out_dim', type=int, default=16,
                      help='Output dimension for node embeddings')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8,
                      help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50,
                      help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                      help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    
    # Subtree parameters
    parser.add_argument('--num_roots', type=int, default=35,
                      help='Number of roots for subtree extraction')
    parser.add_argument('--k_hop', type=int, default=1,
                      help='k-hop for subtree extraction')
    
    # Loss weights
    parser.add_argument('--alpha', type=float, default=10.0,
                      help='Weight for structure loss')
    parser.add_argument('--beta', type=float, default=5.0,
                      help='Weight for weight loss')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='outputs/treegen_asd_lh_35_test',
                      help='Directory to save outputs')
    
    # Weight decay
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay (L2 regularization)')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize modules
    tree_encoder = TreeEncoder(
        in_channels=args.node_feat_dim,
        hidden_channels=args.hidden_dim,
        num_layers=args.num_layers,
        out_channels=args.out_dim
    )
    aggregator = CrossTreeAggregator(
        in_node_dim=args.node_feat_dim,
        in_tree_dim=args.out_dim,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        num_layers=args.num_layers
    )
    decoder = PairwiseDecoder(
        node_dim=args.out_dim,
        hidden_dim=args.hidden_dim
    )
    model = GraphGenModel(tree_encoder, aggregator, decoder).to("cpu")

    # Loss & optimizer
    struct_loss_fn = nn.BCEWithLogitsLoss()
    weight_loss_fn = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Load data based on dataset type
    if args.dataset_type == 'built':
        # Generate synthetic data
        source_data, target_data = load_dataset(
            name='er',
            n_source_nodes=args.n_source_nodes,
            n_target_nodes=args.n_target_nodes,
            n_samples=args.n_samples,
            node_feat_init='adj',
            node_feat_dim=args.node_feat_dim,
            source_edge_prob=args.source_edge_prob,
            target_edge_prob=args.target_edge_prob
        )
    else:  # dataset_type == 'file'
        # Load data from files
        source_data, target_data = load_data(
            src_path=args.src_path,
            trg_path=args.trg_path,
            node_size=args.n_source_nodes,
            feature_strategy=args.feature_strategy
        )

    # Create dataset and split into train/val/test
    full_dataset = GraphPairDataset(
        source_data, target_data,
        m=args.num_roots,
        k=args.k_hop
    )
    
    # Calculate split sizes
    train_size = int(0.6 * len(full_dataset))
    val_size = int(0.2 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    def custom_collate(batch):
        return batch  # returns a list of dicts

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=custom_collate
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate
    )

    # Training for specified number of epochs
    train_losses = {
        'total_loss': [],
        'struct_loss': [],
        'weight_loss': []
    }
    val_losses = {
        'total_loss': [],
        'struct_loss': [],
        'weight_loss': []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    
    print(f"Starting training for {args.num_epochs} epochs...")
    print(f"Training parameters:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Number of roots: {args.num_roots}")
    print(f"  k-hop: {args.k_hop}")
    print(f"  Loss weights - alpha: {args.alpha}, beta: {args.beta}")
    
    for epoch in range(args.num_epochs):
        # Training
        train_epoch_losses = train_epoch(
            model, train_loader, device="cpu",
            struct_loss_fn=struct_loss_fn,
            weight_loss_fn=weight_loss_fn,
            alpha=args.alpha,
            beta=args.beta,
            optimizer=optimizer
        )
        
        # Validation
        val_epoch_losses = validate(
            model, val_loader, device="cpu",
            struct_loss_fn=struct_loss_fn,
            weight_loss_fn=weight_loss_fn,
            alpha=args.alpha,
            beta=args.beta
        )
        
        # Record losses
        for key in train_losses:
            train_losses[key].append(train_epoch_losses[key])
            val_losses[key].append(val_epoch_losses[key])
        
        # Save best model
        if val_epoch_losses['total_loss'] < best_val_loss:
            best_val_loss = val_epoch_losses['total_loss']
            best_model_state = model.state_dict().copy()
        
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        print(f"  Train - Total: {train_epoch_losses['total_loss']:.4f}, "
              f"Struct: {train_epoch_losses['struct_loss']:.4f}, "
              f"Weight: {train_epoch_losses['weight_loss']:.4f}")
        print(f"  Val   - Total: {val_epoch_losses['total_loss']:.4f}, "
              f"Struct: {val_epoch_losses['struct_loss']:.4f}, "
              f"Weight: {val_epoch_losses['weight_loss']:.4f}")
    
    # Load best model for final evaluation
    model.load_state_dict(best_model_state)
    
    # Evaluate on test set
    test_losses = validate(
        model, test_loader, device="cpu",
        struct_loss_fn=struct_loss_fn,
        weight_loss_fn=weight_loss_fn,
        alpha=args.alpha,
        beta=args.beta
    )
    
    print("\nFinal Test Results:")
    print(f"Total Loss: {test_losses['total_loss']:.4f}")
    print(f"Structure Loss: {test_losses['struct_loss']:.4f}")
    print(f"Weight Loss: {test_losses['weight_loss']:.4f}")
    
    # Save losses to numpy file
    np.save(os.path.join(args.output_dir, 'training_losses.npy'), {
        'train': train_losses,
        'val': val_losses,
        'test': test_losses
    })
    
    # Plot and save loss curves
    plot_losses(
        train_losses,
        val_losses,
        save_path=os.path.join(args.output_dir, 'training_losses.png')
    )
    
    # Save best model and configuration
    torch.save({
        'model_state_dict': best_model_state,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_losses': test_losses,
        'args': vars(args)
    }, os.path.join(args.output_dir, 'best_model.pt'))
    
    print("\nTraining completed.")
    print(f"Outputs saved to directory: {args.output_dir}") 