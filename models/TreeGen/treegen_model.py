import torch.nn as nn

class TreeGenModel(nn.Module):
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