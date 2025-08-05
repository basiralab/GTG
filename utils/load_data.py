import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from utils.MatrixVectorizer import MatrixVectorizer
from typing import List, Tuple, Literal

def load_data(
    src_path: str, 
    trg_path: str, 
    node_size: int,
    target_node_size: int = None,
    feature_strategy: Literal["one_hot", "adj", "degree"] = "one_hot"
) -> Tuple[List[Data], List[Data]]:
    """Loads graph data from source and target files and converts them into PyTorch Geometric Data objects.
    
    Args:
        src_path (str): Path to the source graph data file
        trg_path (str): Path to the target graph data file
        node_size (int): Number of nodes in the source graph
        target_node_size (int, optional): Number of nodes in the target graph. If None, uses node_size.
        feature_strategy (str): Strategy for generating node features. One of:
            - "one_hot": One-hot encoding of node indices
            - "adj": Use adjacency matrix rows as features
            - "degree": Use node degrees as features
            
    Returns:
        Tuple[List[Data], List[Data]]: Source and target graph data as PyTorch Geometric Data objects
    """
    if target_node_size is None:
        target_node_size = node_size
        
    # Load and process source data
    src_data = pd.read_csv(src_path, header=None, skiprows=1).drop(columns=[0]).to_numpy()
    src_data_matrix = [MatrixVectorizer.anti_vectorize(row, node_size) for row in src_data]
    src_data_tensor = [torch.tensor(matrix, dtype=torch.float32) for matrix in src_data_matrix]
    
    # Load and process target data
    trg_data = pd.read_csv(trg_path, header=None, skiprows=1).drop(columns=[0]).to_numpy()
    trg_data_matrix = [MatrixVectorizer.anti_vectorize(row, target_node_size) for row in trg_data]
    trg_data_tensor = [torch.tensor(matrix, dtype=torch.float32) for matrix in trg_data_matrix]
    
    def generate_features(adj_matrix: torch.Tensor, strategy: str, size: int) -> torch.Tensor:
        if strategy == "one_hot":
            # One-hot encoding of node indices
            return torch.eye(size, dtype=torch.float32)
        elif strategy == "adj":
            # Use adjacency matrix rows as features
            return adj_matrix
        elif strategy == "degree":
            # Use node degrees as features
            degrees = adj_matrix.sum(dim=1)
            return degrees.unsqueeze(1)
        else:
            raise ValueError(f"Unknown feature strategy: {strategy}")
    
    def create_data_object(adj_matrix: torch.Tensor, features: torch.Tensor, size: int) -> Data:
        # Convert adjacency matrix to edge_index format
        edge_index = torch.nonzero(adj_matrix).t()
        
        return Data(
            x=features,
            edge_index=edge_index,
            num_nodes=size,
            adj=adj_matrix
        )
    
    # Create Data objects for source and target graphs
    src_features_list = [generate_features(adj, feature_strategy, node_size) for adj in src_data_tensor]
    trg_features_list = [generate_features(adj, feature_strategy, target_node_size) for adj in trg_data_tensor]
    
    src_data_list = [create_data_object(adj, feat, node_size) for adj, feat in zip(src_data_tensor, src_features_list)]
    trg_data_list = [create_data_object(adj, feat, target_node_size) for adj, feat in zip(trg_data_tensor, trg_features_list)]
    
    return src_data_list, trg_data_list

if __name__ == "__main__":
    # Example usage
    src_data, trg_data = load_data(
        "single_split_dataset/train_t0.csv",
        "single_split_dataset/train_t1.csv",
        node_size=35,
        target_node_size=70,
        feature_strategy="adj"
    )
    print(len(src_data))
    print(len(trg_data))
    print(f"Source graph shape: {src_data[0].x.shape}")
    print(f"Target graph shape: {trg_data[0].x.shape}")