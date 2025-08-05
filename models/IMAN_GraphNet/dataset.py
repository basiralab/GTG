from torch_geometric.data import Data

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