import torch
import torch.nn as nn

class PairwiseDecoder(nn.Module):
    """
    Pairwise Decoder for graph generation.
    Given node embeddings [N, D], outputs a symmetric adjacency logits matrix and a symmetric edge weight matrix.

    Args:
        node_dim (int): Dimension of node embeddings.
        hidden_dim (int): Hidden dimension for the MLPs.
    """
    def __init__(self, node_dim, hidden_dim):
        super().__init__()
        # Structure branch MLP: predicts edge existence logits
        self.struct_mlp = nn.Sequential(
            nn.Linear(3*node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Weight branch MLP: predicts edge weights
        self.weight_mlp = nn.Sequential(
            nn.Linear(3*node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, h_nodes):
        """
        Args:
            h_nodes (Tensor): [N, D] Node embeddings.
        Returns:
            edge_logits (Tensor): [N, N] Symmetric logits for edge existence (before sigmoid).
            edge_weights (Tensor): [N, N] Symmetric predicted edge weights.
        """
        N, D = h_nodes.size()
        
        # Get indices for upper triangle (excluding diagonal)
        idx_i, idx_j = torch.triu_indices(N, N, offset=1, device=h_nodes.device)
        
        # Concatenate features for upper triangle pairs only
        phi = torch.cat([
            h_nodes[idx_i],                # h_i
            h_nodes[idx_j],                # h_j
            (h_nodes[idx_i]-h_nodes[idx_j]).abs()  # |hi-hj|
        ], dim=1)                          # [M, 3D]

        # Get predictions for upper triangle
        logit = self.struct_mlp(phi).squeeze(-1)   # [M]
        weight = self.weight_mlp(phi).squeeze(-1)  # [M]

        # Initialize full matrices
        logits = torch.zeros(N, N, device=h_nodes.device)
        weights = torch.zeros(N, N, device=h_nodes.device)

        # Fill upper triangle
        logits[idx_i, idx_j] = logit
        weights[idx_i, idx_j] = weight

        # Make symmetric
        logits = logits + logits.t()
        weights = weights + weights.t()

        # No self-loops: set diagonal to -1e9 for logits, 0 for weights
        diag_idx = torch.arange(N, device=h_nodes.device)
        logits[diag_idx, diag_idx] = -1e9
        weights[diag_idx, diag_idx] = 0.0

        return logits, weights
