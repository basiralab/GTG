import torch
from .extract_subtrees import build_subtrees_from_data
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import numpy as np

class TreeEncoder(torch.nn.Module):
    """
    Encodes a list of subtrees (as NetworkX graphs) into fixed-dimensional vectors using GCN and global mean pooling.

    Args:
        in_channels (int): Input feature dimension for each node.
        hidden_channels (int): Hidden dimension for GCN layers.
        num_layers (int): Number of GCN layers.
        out_channels (int, optional): Output embedding dimension. Defaults to hidden_channels.
    """
    def __init__(self, in_channels, hidden_channels, num_layers=2, out_channels=None):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        # First GCN layer: in_channels -> hidden_channels
        self.convs.append(GCNConv(in_channels, hidden_channels))
        # Additional GCN layers: hidden_channels -> hidden_channels
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.out_channels = out_channels or hidden_channels
        # Final linear layer to map to out_channels
        self.fc_out = torch.nn.Linear(hidden_channels, self.out_channels)

    def forward(self, trees, global_node_feats):
        """ 
        Args:
            trees (list of nx.Graph): Each subtree as a NetworkX graph.
            global_node_feats (Tensor): [N, F] Node features for the full graph.
        Returns:
            z (Tensor): [m, D] Embedding for each subtree (m = number of subtrees, D = out_channels)
        """
        data_list = []
        for tree in trees:
            nodes = list(tree.nodes())
            # Extract features for nodes in the subtree
            x = global_node_feats[nodes]
            # Build edge_index for PyG (bidirectional)
            edge_idx = []
            for u, v in tree.edges():
                ui, vi = nodes.index(u), nodes.index(v)
                edge_idx.append([ui, vi])
                edge_idx.append([vi, ui])
            if edge_idx:
                edge_index = torch.tensor(edge_idx, dtype=torch.long).t().contiguous()
            else:
                edge_index = torch.empty((2,0), dtype=torch.long)
            data_list.append(Data(x=x, edge_index=edge_index))
        # Batch all subtrees for efficient processing
        loader = DataLoader(data_list, batch_size=len(data_list), shuffle=False)
        batch = next(iter(loader))
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
        # Apply GCN layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
        # Global mean pooling to get one embedding per subtree
        z = global_mean_pool(x, batch_idx)
        # Final linear projection
        z = self.fc_out(z)
        return z
