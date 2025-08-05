import torch
from .extract_subtrees import build_subtrees_from_data
from torch.utils.data import Dataset
import networkx as nx

class TreeGenDataset(Dataset):
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
        target_struct = (target_weight > 0).float()
        
        return {
            "G": G,
            "global_feats": global_feats,
            "subtrees_nodes": subtrees_nodes,
            "target_struct": target_struct,
            "target_weight": target_weight
        }

    def __len__(self):
        return len(self.source_data)