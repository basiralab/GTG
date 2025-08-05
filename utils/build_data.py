import torch
import numpy as np
import networkx as nx
from functools import partial
from torch_geometric.data import Data

# --- Graph Construction Functions ---

def construct_er_dataset(n_source_nodes, n_target_nodes, n_samples, source_edge_prob, target_edge_prob):
    source_mat_all = [create_er_graph(n_source_nodes, source_edge_prob) for _ in range(n_samples)]
    target_mat_all = [create_er_graph(n_target_nodes, target_edge_prob) for _ in range(n_samples)]
    return source_mat_all, target_mat_all

def construct_ba_dataset(n_source_nodes, n_target_nodes, n_samples, n_source_edges_per_node, n_target_edges_per_node):
    source_mat_all = [create_ba_graph(n_source_nodes, n_source_edges_per_node) for _ in range(n_samples)]
    target_mat_all = [create_ba_graph(n_target_nodes, n_target_edges_per_node) for _ in range(n_samples)]
    return source_mat_all, target_mat_all

def construct_kronecker_dataset(n_source_nodes, n_target_nodes, n_samples, source_init_matrix_size, target_init_matrix_size, n_iterations):
    assert n_source_nodes == source_init_matrix_size ** n_iterations, "Number of nodes in source graph does not match size of Kronecker graph"
    assert n_target_nodes == target_init_matrix_size ** n_iterations, "Number of nodes in target graph does not match size of Kronecker graph"
    source_mat_all = [create_kronecker_graph(source_init_matrix_size, n_iterations) for _ in range(n_samples)]
    target_mat_all = [create_kronecker_graph(target_init_matrix_size, n_iterations) for _ in range(n_samples)]
    return source_mat_all, target_mat_all

def construct_sbm_dataset(n_source_nodes, n_target_nodes, n_samples, source_blocks, target_blocks, source_P, target_P):
    assert n_source_nodes == sum(source_blocks), "Number of nodes in source graph does not match sum of nodes in blocks"
    assert n_target_nodes == sum(target_blocks), "Number of nodes in target graph does not match sum of nodes in blocks"
    source_P = np.array(source_P)
    target_P = np.array(target_P)
    source_mat_all = [create_sbm_graph(source_blocks, source_P) for _ in range(n_samples)]
    target_mat_all = [create_sbm_graph(target_blocks, target_P) for _ in range(n_samples)]
    return source_mat_all, target_mat_all

# --- Main Loader ---

def load_dataset(
    name,
    n_source_nodes,
    n_target_nodes,
    n_samples,
    node_feat_init,
    node_feat_dim,
    source_edge_prob=None,
    target_edge_prob=None,
    n_source_edges_per_node=None,
    n_target_edges_per_node=None,
    source_init_matrix_size=None,
    target_init_matrix_size=None,
    n_iterations=None,
    source_blocks=None,
    target_blocks=None,
    source_P=None,
    target_P=None
):
    if name == 'er':
        source_mat_all, target_mat_all = construct_er_dataset(
            n_source_nodes, n_target_nodes, n_samples, source_edge_prob, target_edge_prob
        )
    elif name == 'ba':
        source_mat_all, target_mat_all = construct_ba_dataset(
            n_source_nodes, n_target_nodes, n_samples, n_source_edges_per_node, n_target_edges_per_node
        )
    elif name == 'kronecker':
        source_mat_all, target_mat_all = construct_kronecker_dataset(
            n_source_nodes, n_target_nodes, n_samples, source_init_matrix_size, target_init_matrix_size, n_iterations
        )
    elif name == 'sbm':
        source_mat_all, target_mat_all = construct_sbm_dataset(
            n_source_nodes, n_target_nodes, n_samples, source_blocks, target_blocks, source_P, target_P
        )
    else:
        raise ValueError(f"Unsupported dataset type: {name}")

    # Convert to torch tensors
    source_mat_all = [torch.tensor(x, dtype=torch.float) for x in source_mat_all]
    target_mat_all = [torch.tensor(x, dtype=torch.float) for x in target_mat_all]

    # Convert to PyG
    pyg_partial = partial(create_pyg_graph, node_feature_init=node_feat_init, node_feat_dim=node_feat_dim)
    source_pyg_all = [pyg_partial(x, n_source_nodes) for x in source_mat_all]
    target_pyg_all = [pyg_partial(x, n_target_nodes) for x in target_mat_all]

    # Prepare source and target data
    source_data = [{'pyg': source_pyg, 'mat': source_mat} for source_pyg, source_mat in zip(source_pyg_all, source_mat_all)]
    target_data = [{'pyg': target_pyg, 'mat': target_mat} for target_pyg, target_mat in zip(target_pyg_all, target_mat_all)]

    return source_data, target_data

# --- Graph Generators (unchanged) ---

def create_er_graph(n_nodes, edge_prob):
    G = nx.erdos_renyi_graph(n_nodes, edge_prob)
    adj = nx.adjacency_matrix(G).toarray()
    return adj

def create_ba_graph(n_nodes, n_edges):
    G = nx.barabasi_albert_graph(n_nodes, n_edges)
    adj = nx.adjacency_matrix(G).toarray()
    return adj

def create_symmetric_initiator_matrix(size, low=0.5, high=1.0, diagonal_value=0.1):
    matrix = np.random.uniform(low, high, (size, size))
    symmetric_matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(symmetric_matrix, diagonal_value)
    return symmetric_matrix

def kronecker_product(init_matrix, iterations):
    result = init_matrix
    for _ in range(iterations - 1):
        result = np.kron(result, init_matrix)
    return result

def create_kronecker_graph(init_matrix_size, iterations):
    init_matrix = create_symmetric_initiator_matrix(init_matrix_size, 0.5, 1.0)
    adj_matrix = kronecker_product(init_matrix, iterations)
    G = nx.from_numpy_array((adj_matrix > np.random.rand(*adj_matrix.shape)).astype(int))
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        for i in range(len(components) - 1):
            u = list(components[i])[0]
            v = list(components[i+1])[0]
            G.add_edge(u, v)
    adj = nx.adjacency_matrix(G).toarray()
    return adj

def create_sbm_graph(block_sizes, P):
    N = sum(block_sizes)
    G = nx.Graph()
    G.add_nodes_from(range(N))
    block_membership = []
    current_node = 0
    for block_id, size in enumerate(block_sizes):
        block_membership.extend([block_id] * size)
        current_node += size
    for i in range(N):
        for j in range(i + 1, N):
            block_i = block_membership[i]
            block_j = block_membership[j]
            prob_edge = P[block_i, block_j]
            if np.random.rand() < prob_edge:
                G.add_edge(i, j)
    adj = nx.adjacency_matrix(G).toarray()
    return adj

def create_pyg_graph(x, n_nodes, node_feature_init='adj', node_feat_dim=1):
    if isinstance(x, torch.Tensor):
        edge_attr = x.view(-1, 1)
    else:
        edge_attr = torch.tensor(x, dtype=torch.float).view(-1, 1)
    if node_feature_init == 'adj':
        if isinstance(x, torch.Tensor):
            node_feat = x
        else:
            node_feat = torch.tensor(x, dtype=torch.float)
    elif node_feature_init == 'random':
        node_feat = torch.randn(n_nodes, node_feat_dim, device=edge_attr.device)
    elif node_feature_init == 'ones':
        node_feat = torch.ones(n_nodes, node_feat_dim, device=edge_attr.device)
    else:
        raise ValueError(f"Unsupported node feature initialization: {node_feature_init}")
    
    # Create edge_index from adjacency matrix
    rows, cols = torch.where(torch.tensor(x) > 0)
    edge_index = torch.stack([rows, cols], dim=0)
    
    pyg_graph = Data(x=node_feat, edge_index=edge_index, edge_attr=edge_attr)
    return pyg_graph

if __name__ == "__main__":
    # For ER
    source_data, target_data = load_dataset(
        name='er',
        n_source_nodes=20,
        n_target_nodes=20,
        n_samples=100,
        node_feat_init='adj',
        node_feat_dim=1,
        source_edge_prob=0.2,
        target_edge_prob=0.3
    )

    # For BA
    source_data, target_data = load_dataset(
        name='ba',
        n_source_nodes=20,
        n_target_nodes=20,
        n_samples=100,
        node_feat_init='adj',
        node_feat_dim=1,
        n_source_edges_per_node=2,
        n_target_edges_per_node=3
    )

    # For Kronecker
    source_data, target_data = load_dataset(
        name='kronecker',
        n_source_nodes=16,
        n_target_nodes=16,
        n_samples=100,
        node_feat_init='adj',
        node_feat_dim=1,
        source_init_matrix_size=2,
        target_init_matrix_size=2,
        n_iterations=4
    )

    # For SBM
    source_data, target_data = load_dataset(
        name='sbm',
        n_source_nodes=20,
        n_target_nodes=20,
        n_samples=100,
        node_feat_init='adj',
        node_feat_dim=1,
        source_blocks=[10, 10],
        target_blocks=[10, 10],
        source_P=[[0.8, 0.2], [0.2, 0.7]],
        target_P=[[0.7, 0.3], [0.3, 0.6]]
    )