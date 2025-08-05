from utils.MatrixVectorizer import MatrixVectorizer
from sklearn.metrics import mean_absolute_error
import torch
import networkx as nx
import numpy as np
from tqdm import tqdm as tqdm_base
from tqdm.notebook import tqdm as tqdm_notebook
from networkx.algorithms.centrality import katz_centrality_numpy, laplacian_centrality, information_centrality

def laplacian_frobenius_distance(adj_true, adj_pred):
    """
    Frobenius distance of Laplacian:
    Measures global discrepancy in connectivity and node degree distribution.
    """
    L_true = np.diag(np.sum(adj_true, axis=1)) - adj_true
    L_pred = np.diag(np.sum(adj_pred, axis=1)) - adj_pred
    return np.linalg.norm(L_true - L_pred, ord='fro')

def clustering_coefficient_difference(true_adj, pred_adj):
    """
    Mean absolute difference in clustering coefficient:
    Measures local triangle/cluster preservation between graphs.
    """
    true_graph = nx.from_numpy_array(true_adj)
    pred_graph = nx.from_numpy_array(pred_adj)
    true_vals = np.array(list(nx.clustering(true_graph, weight='weight').values()))
    pred_vals = np.array(list(nx.clustering(pred_graph, weight='weight').values()))
    return np.mean(np.abs(true_vals - pred_vals))

def safe_centrality_computation(graph, centrality_func, **kwargs):
    """
    Safely compute centrality measures with error handling for disconnected graphs.
    """
    try:
        return centrality_func(graph, **kwargs)
    except (nx.NetworkXError, ValueError, np.linalg.LinAlgError) as e:
        # If the centrality fails, return uniform centrality values
        n_nodes = len(graph.nodes())
        return {node: 1.0/n_nodes for node in graph.nodes()}

def evaluate_single_sample(pred_matrix, gt_matrix):
    # Support path, Tensor, and ndarray input
    if isinstance(pred_matrix, str):
        pred_matrix = np.load(pred_matrix)
    if isinstance(gt_matrix, str):
        gt_matrix = np.load(gt_matrix)
    if isinstance(pred_matrix, torch.Tensor):
        pred_matrix = pred_matrix.numpy()
    if isinstance(gt_matrix, torch.Tensor):
        gt_matrix = gt_matrix.numpy()

    # Build weighted graph
    pred_graph = nx.from_numpy_array(pred_matrix, edge_attr="weight")
    gt_graph   = nx.from_numpy_array(gt_matrix,   edge_attr="weight")

    # 0. Overall MAE: mean absolute error of all centrality metrics
    pred_vec = MatrixVectorizer.vectorize(pred_matrix)
    gt_vec   = MatrixVectorizer.vectorize(gt_matrix)
    mae      = mean_absolute_error(pred_vec, gt_vec)

    # 1. Degree centrality: fraction of possible edges attached to each node
    deg_pred = nx.degree_centrality(pred_graph)
    deg_gt   = nx.degree_centrality(gt_graph)
    mae_deg  = mean_absolute_error(list(deg_pred.values()), list(deg_gt.values()))

    # 2. Betweenness centrality: fraction of shortest paths going through each node
    bc_pred  = safe_centrality_computation(pred_graph, nx.betweenness_centrality, weight='weight')
    bc_gt    = safe_centrality_computation(gt_graph, nx.betweenness_centrality, weight='weight')
    mae_bc   = mean_absolute_error(list(bc_pred.values()), list(bc_gt.values()))

    # 3. Eigenvector centrality: influence via connections to high-centrality nodes
    ec_pred  = safe_centrality_computation(pred_graph, nx.eigenvector_centrality, weight='weight')
    ec_gt    = safe_centrality_computation(gt_graph, nx.eigenvector_centrality, weight='weight')
    mae_ec   = mean_absolute_error(list(ec_pred.values()), list(ec_gt.values()))

    # 4. Information centrality: centrality based on information flow efficiency
    ic_pred  = safe_centrality_computation(pred_graph, information_centrality, weight='weight')
    ic_gt    = safe_centrality_computation(gt_graph, information_centrality, weight='weight')
    mae_ic   = mean_absolute_error(list(ic_pred.values()), list(ic_gt.values()))

    # 5. PageRank: random-walk based centrality with damping
    pr_pred  = safe_centrality_computation(pred_graph, nx.pagerank, weight='weight')
    pr_gt    = safe_centrality_computation(gt_graph, nx.pagerank, weight='weight')
    mae_pr   = mean_absolute_error(list(pr_pred.values()), list(pr_gt.values()))

    # 6. Katz centrality: counts all walks with attenuation factor
    katz_pred = safe_centrality_computation(pred_graph, katz_centrality_numpy, weight='weight')
    katz_gt   = safe_centrality_computation(gt_graph, katz_centrality_numpy, weight='weight')
    mae_katz  = mean_absolute_error(list(katz_pred.values()), list(katz_gt.values()))

    # 7. Laplacian centrality: change in Laplacian energy when removing a node
    lap_pred  = safe_centrality_computation(pred_graph, laplacian_centrality)
    lap_gt    = safe_centrality_computation(gt_graph, laplacian_centrality)
    mae_lap   = mean_absolute_error(list(lap_pred.values()), list(lap_gt.values()))

    # Additional structural metrics
    clustering_diff = clustering_coefficient_difference(gt_matrix, pred_matrix)
    laplacian_dist  = laplacian_frobenius_distance(gt_matrix, pred_matrix)

    return {
        "mae": mae,
        "mae_deg": mae_deg,
        "mae_bc": mae_bc,
        "mae_ec": mae_ec,
        "mae_ic": mae_ic,
        "mae_pr": mae_pr,
        "mae_katz": mae_katz,
        "mae_lap": mae_lap,
        "clustering_diff": clustering_diff,
        "laplacian_frobenius_distance": laplacian_dist
    }

def evaluate(pred_matrices, gt_matrices, show_progress=True, from_notebook=False):
    # Support path, Tensor, and ndarray input
    if isinstance(pred_matrices, str):
        pred_matrices = np.load(pred_matrices)
    if isinstance(gt_matrices, str):
        gt_matrices = np.load(gt_matrices)
    if isinstance(pred_matrices, torch.Tensor):
        pred_matrices = pred_matrices.numpy()
    if isinstance(gt_matrices, torch.Tensor):
        gt_matrices = gt_matrices.numpy()

    # Prepare lists for each metric
    mae_lists = {key: [] for key in ["mae","mae_deg","mae_bc","mae_ec","mae_ic","mae_pr","mae_katz","mae_lap"]}
    clustering_list = []
    laplacian_list = []

    tq = tqdm_notebook if from_notebook else tqdm_base
    for i in tq(range(len(pred_matrices)), desc="Evaluating", disable=not show_progress):
        try:
            res = evaluate_single_sample(pred_matrices[i], gt_matrices[i])
            for key in mae_lists:
                mae_lists[key].append(res[key])
            clustering_list.append(res["clustering_diff"])
            laplacian_list.append(res["laplacian_frobenius_distance"])
        except Exception as e:
            print(f"Warning: Failed to evaluate sample {i}: {str(e)}")
            # Add default values for failed evaluation
            for key in mae_lists:
                mae_lists[key].append(1.0)  # Default high error value
            clustering_list.append(1.0)
            laplacian_list.append(1.0)

    # Summary
    summary = {key: np.mean(mae_lists[key]) for key in mae_lists}
    summary["clustering_diff"] = np.mean(clustering_list)
    summary["laplacian_frobenius_distance"] = np.mean(laplacian_list)
    return summary