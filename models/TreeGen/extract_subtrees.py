import torch
import numpy as np
import networkx as nx
from collections import deque
from utils import load_dataset

# --------------------------------------------
# 1. Entropy-based root selection
# --------------------------------------------
def compute_node_entropy(G: nx.Graph) -> dict:
    entropies = {}
    for v in G.nodes():
        sub_nodes = list(G.neighbors(v)) + [v]
        degs = np.array([G.degree(u) for u in sub_nodes], dtype=float)
        deg_sum = degs.sum()
        if deg_sum > 0:
            probs = degs / deg_sum
        else:
            # If all degrees are zero, use uniform probabilities
            probs = np.ones_like(degs) / len(degs)
        entropies[v] = -np.sum(probs * np.log2(probs + 1e-12))
    return entropies

def select_roots_by_entropy(G: nx.Graph, m: int) -> list:
    ent = compute_node_entropy(G)
    roots = sorted(ent, key=lambda v: -ent[v])[:m]
    return roots

# --------------------------------------------
# 2. Triangle motif-based root selection
# --------------------------------------------
def find_triangles(G: nx.Graph) -> list:
    triangles = []
    for clique in nx.enumerate_all_cliques(G):
        if len(clique) == 3:
            triangles.append(tuple(clique))
    return triangles

def select_roots_by_triangle(G: nx.Graph, m: int) -> list:
    counts = {v: 0 for v in G.nodes()}
    for tri in find_triangles(G):
        for v in tri:
            counts[v] += 1
    roots = sorted(counts, key=lambda v: -counts[v])[:m]
    return roots

# --------------------------------------------
# 3. k-hop BFS subtree extraction
# --------------------------------------------
def extract_k_hop_subtree(G: nx.Graph, root: int, k: int) -> nx.Graph:
    visited = {root}
    tree_edges = []
    queue = deque([(root, 0)])
    while queue:
        node, depth = queue.popleft()
        if depth < k:
            for nbr in G.neighbors(node):
                if nbr not in visited:
                    visited.add(nbr)
                    tree_edges.append((node, nbr))
                    queue.append((nbr, depth + 1))
    T = nx.Graph()
    T.add_nodes_from(visited)
    T.add_edges_from(tree_edges)
    return T

# --------------------------------------------
# 4. Main: apply to built data
# --------------------------------------------
def build_subtrees_from_data(data_list, m: int, k: int, root_strategy='entropy'):
    """
    data_list: list of dicts, each with 'pyg' and 'mat' (from build_data.py)
    m: number of roots/subtrees per graph
    k: k-hop for subtree
    root_strategy: 'entropy' or 'triangle'
    Returns: list of list of subtrees (per graph)
    """
    all_subtrees = []
    for idx, data in enumerate(data_list):
        adj = data['mat'].cpu().numpy() if torch.is_tensor(data['mat']) else data['mat']
        G = nx.from_numpy_array(adj)
        if root_strategy == 'entropy':
            roots = select_roots_by_entropy(G, m)
        elif root_strategy == 'triangle':
            roots = select_roots_by_triangle(G, m)
        else:
            raise ValueError("Unknown root_strategy")
        subtrees = [extract_k_hop_subtree(G, r, k) for r in roots]
        all_subtrees.append(subtrees)
    return all_subtrees
