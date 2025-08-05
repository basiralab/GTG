import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import TransformerConv, global_mean_pool, GCNConv, GATConv
import torch_geometric
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from utils import load_data, GraphTreeProducer, WLConfig
from utils.build_data import load_dataset

# 1. Dataset Preparation
class ExemplarGraphDataset(Dataset):
    def __init__(self, src_data_list, trg_data_list, wl_config, k=10, frac_to_sample=1.0):
        self.samples = []
        for i, (src_data, trg_data) in enumerate(zip(src_data_list, trg_data_list)):
            # Handle both data formats
            if isinstance(src_data, dict):
                # Synthetic data format from build_data.py
                src_pyg = src_data['pyg']
                target_adj = trg_data['mat']
            else:
                # Custom loaded data format
                src_pyg = src_data
                target_adj = trg_data.adj

            src_producer = GraphTreeProducer(src_pyg, wl_config=wl_config)
            exemplars = src_producer.sample_and_select_exemplars(
                k=k, frac_to_sample=frac_to_sample, metric="cosine"
            )
            # print(f"Sample {i}: Got {len(exemplars)} exemplars (requested k={k})")
            wl_reprs = src_producer.compute_wl_representations()
            exemplar_wl = wl_reprs[exemplars]  # [num_exemplars, F]
            self.samples.append((exemplar_wl, target_adj))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        exemplar_wl, target_adj = self.samples[idx]
        num_exemplars = exemplar_wl.shape[0]
        # print(f"Getting item {idx}: {num_exemplars} exemplars")
        # For GNNs, use fully connected edge_index
        edge_index = torch.combinations(torch.arange(num_exemplars), r=2).t()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        return exemplar_wl, edge_index, target_adj

# 2. Model Definition
class Exemplar2AdjModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_exemplars, out_nodes=35, num_layers=2):
        super().__init__()
        self.num_exemplars = num_exemplars
        self.out_nodes = out_nodes
        self.hidden_dim = hidden_dim

        # Use TransformerConv or GCNConv, but since we don't have a graph structure among exemplars,
        # we can use MLP or treat exemplars as a fully connected graph.
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # Pool all exemplar features into a single graph embedding
        self.pool = nn.AdaptiveAvgPool1d(1)
        # Decoder: map pooled embedding to flattened adjacency
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_nodes * out_nodes)
        )

    def forward(self, x):
        # x: [num_exemplars, in_dim]
        x = self.encoder(x)  # [num_exemplars, hidden_dim]
        x = x.transpose(0, 1).unsqueeze(0)  # [1, hidden_dim, num_exemplars]
        pooled = self.pool(x).squeeze()  # [hidden_dim]
        out = self.decoder(pooled)  # [out_nodes * out_nodes]
        out = out.view(self.out_nodes, self.out_nodes)  # [out_nodes, out_nodes]
        return out
    
class ExemplarGNN2AdjModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_exemplars, out_nodes=35, model_type="mlp", heads=2, pairwise_decoder=True):
        super().__init__()
        self.num_exemplars = num_exemplars
        self.out_nodes = out_nodes
        self.hidden_dim = hidden_dim
        self.model_type = model_type
        self.pairwise_decoder = pairwise_decoder

        if model_type == "mlp":
            self.encoder = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
        elif model_type == "gcn":
            self.gnn1 = GCNConv(in_dim, hidden_dim)
            self.gnn2 = GCNConv(hidden_dim, hidden_dim)
        elif model_type == "gat":
            self.gnn1 = GATConv(in_dim, hidden_dim, heads=heads, concat=True)
            self.gnn2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=True)
        elif model_type == "trans":
            self.gnn1 = TransformerConv(in_dim, hidden_dim, heads=heads)
            self.gnn2 = TransformerConv(hidden_dim * heads, hidden_dim, heads=1)
        else:
            raise ValueError("Unknown model_type")

        if pairwise_decoder:
            self.pairwise_mlp = nn.Sequential(
                nn.Linear(3 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        else:
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_nodes * out_nodes)
            )

    def forward(self, x, edge_index=None):
        if self.model_type == "mlp":
            x = self.encoder(x)
        else:
            x = self.gnn1(x, edge_index)
            x = torch.relu(x)
            x = self.gnn2(x, edge_index)
            x = torch.relu(x)
        # x: [b, d]
        if self.pairwise_decoder:
            b, d = x.shape
            # print(f"Model input shape: {x.shape}, target shape should be [{self.out_nodes}, {self.out_nodes}]")
            hi = x.unsqueeze(1).repeat(1, b, 1)  # [b, b, d]
            hj = x.unsqueeze(0).repeat(b, 1, 1)  # [b, b, d]
            diff = (hi - hj).abs()               # [b, b, d]
            feat = torch.cat([hi, hj, diff], dim=-1)  # [b, b, 3d]
            F = feat.view(b * b, 3 * d)               # [b^2, 3d]
            scores = self.pairwise_mlp(F).view(b, b)  # [b, b]
            # print(f"Model output shape: {scores.shape}")
            return scores
        else:
            x = x.transpose(0, 1).unsqueeze(0)  # [1, hidden_dim, num_exemplars]
            pooled = self.pool(x).squeeze()     # [hidden_dim]
            out = self.decoder(pooled)
            out = out.view(self.out_nodes, self.out_nodes)
            return out

def split_indices(n, val_ratio=0.1, test_ratio=0.1, seed=42):
    idx = np.arange(n)
    train_idx, test_idx = train_test_split(idx, test_size=test_ratio, random_state=seed)
    train_idx, val_idx = train_test_split(train_idx, test_size=val_ratio/(1-test_ratio), random_state=seed)
    return train_idx, val_idx, test_idx

# 3. Training Loop
def train():
    # ------------------- Hyperparameters -------------------
    node_size = 35
    feature_strategy = "adj"
    wl_iterations = 2
    k = 10
    frac_to_sample = 1.0
    hidden_dim = 64
    num_epochs = 50
    batch_size = 1
    learning_rate = 1e-3
    val_ratio = 0.1
    test_ratio = 0.1
    seed = 42
    model_type = "mlp"  # Try "mlp", "gcn", "gat", or "trans"
    best_model_path = f"best_exemplar2adj_model_{model_type}.pt"
    early_stopping_patience = 10
    early_stopping_delta = 1e-5
    loss_curve_path = f"loss_curve_{model_type}.png"
    loss_type = "bce"  # Options: "bce", "mse", "l1", or combinations
    # -------------------------------------------------------

    # --- Configurable flag ---
    use_synthetic = False  # Set to False to use CSVs

    node_size = 35
    n_samples = 200

    if use_synthetic:
        src_data_list, trg_data_list = load_dataset(
            name='er',
            n_source_nodes=node_size,
            n_target_nodes=node_size,
            n_samples=n_samples,
            node_feat_init='adj',
            node_feat_dim=35,
            source_edge_prob=0.2,
            target_edge_prob=0.3
        )
    else:
        src_path = "single_split_dataset/train_t0.csv"
        trg_path = "single_split_dataset/train_t0.csv"
        feature_strategy = "adj"
        src_data_list, trg_data_list = load_data(
            src_path,
            trg_path,
            node_size=node_size,
            feature_strategy=feature_strategy
        )
    wl_config = WLConfig(add_self_loops=True, normalize_adj=True, iterations=wl_iterations, symmetric=True)

    # Split indices
    n = len(src_data_list)
    train_idx, val_idx, test_idx = split_indices(n, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)
    train_src, train_trg = [src_data_list[i] for i in train_idx], [trg_data_list[i] for i in train_idx]
    val_src, val_trg = [src_data_list[i] for i in val_idx], [trg_data_list[i] for i in val_idx]
    test_src, test_trg = [src_data_list[i] for i in test_idx], [trg_data_list[i] for i in test_idx]

    # Get input dimension from first sample
    if isinstance(train_src[0], dict):
        in_dim = train_src[0]['pyg'].x.shape[1]
    else:
        in_dim = train_src[0].x.shape[1]

    # Datasets and loaders
    train_dataset = ExemplarGraphDataset(train_src, train_trg, wl_config, k=k, frac_to_sample=frac_to_sample)
    val_dataset = ExemplarGraphDataset(val_src, val_trg, wl_config, k=k, frac_to_sample=frac_to_sample)
    test_dataset = ExemplarGraphDataset(test_src, test_trg, wl_config, k=k, frac_to_sample=frac_to_sample)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model
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
    if loss_type == "bce":
        criterion = nn.BCEWithLogitsLoss()
    elif loss_type == "mse":
        criterion = nn.MSELoss()
    elif loss_type == "l1":
        criterion = nn.L1Loss()
    elif loss_type == "bce+mse":
        criterion_bce = nn.BCEWithLogitsLoss()
        criterion_mse = nn.MSELoss()
    elif loss_type == "all":
        criterion_bce = nn.BCEWithLogitsLoss()
        criterion_mse = nn.MSELoss()
        criterion_l1 = nn.L1Loss()

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for exemplar_wl, edge_index, target_adj in train_bar:
            exemplar_wl = exemplar_wl.squeeze(0)
            edge_index = edge_index.squeeze(0)
            target_adj = target_adj.squeeze(0)
            if model_type == "mlp":
                pred_adj = model(exemplar_wl)
            else:
                pred_adj = model(exemplar_wl, edge_index)
            if loss_type == "bce":
                loss = criterion(pred_adj, target_adj)
            elif loss_type == "mse":
                loss = criterion(pred_adj, target_adj)
            elif loss_type == "l1":
                loss = criterion(pred_adj, target_adj)
            elif loss_type == "bce+mse":
                loss = criterion_bce(pred_adj, target_adj) + criterion_mse(torch.sigmoid(pred_adj), target_adj)
            elif loss_type == "all":
                loss = criterion_bce(pred_adj, target_adj) + criterion_mse(pred_adj, target_adj) + criterion_l1(pred_adj, target_adj)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False)
            for exemplar_wl, edge_index, target_adj in val_bar:
                exemplar_wl = exemplar_wl.squeeze(0)
                edge_index = edge_index.squeeze(0)
                target_adj = target_adj.squeeze(0)
                if model_type == "mlp":
                    pred_adj = model(exemplar_wl)
                else:
                    pred_adj = model(exemplar_wl, edge_index)
                if loss_type == "bce":
                    loss = criterion(pred_adj, target_adj)
                elif loss_type == "mse":
                    loss = criterion(pred_adj, target_adj)
                elif loss_type == "l1":
                    loss = criterion(pred_adj, target_adj)
                elif loss_type == "bce+mse":
                    loss = criterion_bce(pred_adj, target_adj) + criterion_mse(torch.sigmoid(pred_adj), target_adj)
                elif loss_type == "all":
                    loss = criterion_bce(pred_adj, target_adj) + criterion_mse(pred_adj, target_adj) + criterion_l1(pred_adj, target_adj)
                val_loss += loss.item()
                val_bar.set_postfix(loss=loss.item())
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        tqdm.write(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f}")

        # Early stopping logic
        if avg_val_loss < best_val_loss - early_stopping_delta:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            tqdm.write(f"Best model saved at epoch {epoch} with val loss {best_val_loss:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                tqdm.write(f"Early stopping at epoch {epoch} (no improvement for {early_stopping_patience} epochs)")
                break

    print(f"Training complete. Best model saved to {best_model_path}")

    # --- TEST PHASE ---
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    test_loss = 0
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc="Test", leave=False)
        for exemplar_wl, edge_index, target_adj in test_bar:
            exemplar_wl = exemplar_wl.squeeze(0)
            edge_index = edge_index.squeeze(0)
            target_adj = target_adj.squeeze(0)
            if model_type == "mlp":
                pred_adj = model(exemplar_wl)
            else:
                pred_adj = model(exemplar_wl, edge_index)
            if loss_type == "bce":
                loss = criterion(pred_adj, target_adj)
            elif loss_type == "mse":
                loss = criterion(pred_adj, target_adj)
            elif loss_type == "l1":
                loss = criterion(pred_adj, target_adj)
            elif loss_type == "bce+mse":
                loss = criterion_bce(pred_adj, target_adj) + criterion_mse(torch.sigmoid(pred_adj), target_adj)
            elif loss_type == "all":
                loss = criterion_bce(pred_adj, target_adj) + criterion_mse(pred_adj, target_adj) + criterion_l1(pred_adj, target_adj)
            test_loss += loss.item()
            test_bar.set_postfix(loss=loss.item())
    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")

    # Plot and save loss curves
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss ({model_type})')
    plt.legend()
    plt.savefig(loss_curve_path)
    print(f"Loss curve saved to {loss_curve_path}")
    plt.close()

if __name__ == "__main__":
    train()