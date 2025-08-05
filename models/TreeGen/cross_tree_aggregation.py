import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from .extract_subtrees import build_subtrees_from_data
from utils.build_data import load_dataset

class CrossTreeAggregator(nn.Module):
    """
    Bipartite GNN for cross-aggregation between subtrees and original graph nodes.
    Now supports super-resolution by handling different input/output dimensions.
    """
    def __init__(self, in_node_dim, in_tree_dim, hidden_dim, out_dim, num_layers=2, 
                 output_nodes=None, upsampling_factor=1):
        super().__init__()
        self.in_node_dim = in_node_dim
        self.in_tree_dim = in_tree_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.output_nodes = output_nodes
        self.upsampling_factor = upsampling_factor
        
        # Projections for input nodes and trees
        self.node_proj = nn.Linear(in_node_dim, hidden_dim)
        self.tree_proj = nn.Linear(in_tree_dim, hidden_dim)
        
        # GNN layers
        self.convs = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # Output projections
        self.node_out = nn.Linear(hidden_dim, out_dim)
        self.tree_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, global_node_feats, z_trees, subtrees_nodes):
        N, Fn = global_node_feats.shape
        m, Ft = z_trees.shape
        
        # Project input features
        X = torch.zeros((N + m, self.hidden_dim), device=global_node_feats.device)
        X[:N] = self.node_proj(global_node_feats)
        X[N:] = self.tree_proj(z_trees)
        X = torch.relu(X)

        # Build bipartite edge_index
        src, dst = [], []
        for k, nodes in enumerate(subtrees_nodes):
            tree_id = N + k
            for v in nodes:
                src.append(tree_id); dst.append(v)
                src.append(v);       dst.append(tree_id)
        edge_index = torch.tensor([src, dst], dtype=torch.long,
                                  device=global_node_feats.device)

        # Apply GNN layers
        for conv in self.convs:
            X = conv(X, edge_index)
            X = torch.relu(X)

        # Handle super-resolution
        if self.output_nodes is not None:
            # Get node features
            h_nodes = X[:N]  # [N, hidden_dim]
            
            # Project to output dimension first
            h_nodes = self.node_out(h_nodes)  # [N, out_dim]
            
            # Reshape for interpolation: [N, out_dim] -> [1, out_dim, N, 1]
            h_nodes = h_nodes.unsqueeze(0).unsqueeze(-1).transpose(1, 2)
            
            # Interpolate to target size
            h_nodes = torch.nn.functional.interpolate(
                h_nodes,  # [1, out_dim, N, 1]
                size=(self.output_nodes, 1),  # Interpolate only first dim
                mode='bilinear',
                align_corners=True
            )
            
            # Reshape back: [1, out_dim, output_nodes, 1] -> [output_nodes, out_dim]
            h_nodes = h_nodes.squeeze(0).squeeze(-1).transpose(0, 1)
            
            # Update tree embeddings
            z_trees_updated = self.tree_out(X[N:])
            
            return h_nodes, z_trees_updated
        else:
            # Original behavior for same-size graphs
            h_nodes = self.node_out(X[:N])
        z_trees_updated = self.tree_out(X[N:])
        return h_nodes, z_trees_updated

def check_subtree_coverage(subtrees_nodes, num_nodes):
    """
    Check if the union of all subtrees' nodes covers all nodes in the graph.
    """
    covered = set()
    for nodes in subtrees_nodes:
        covered.update(nodes)
    return len(covered) == num_nodes
