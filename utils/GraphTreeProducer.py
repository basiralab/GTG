import numpy as np
import scipy.sparse as sp
import torch
from torch_sparse import SparseTensor
from torch_sparse import matmul as sparse_matmul
from torch_geometric.data import Data
from typing import Literal, Tuple, Dict, Optional, List, Union
from dataclasses import dataclass
from collections import defaultdict
import heapq
from tqdm import tqdm

@dataclass
class WLConfig:
    """Configuration for WL representation computation"""
    add_self_loops: bool = True
    normalize_adj: bool = True
    iterations: int = 1
    symmetric: bool = True

class GraphTreeProducer:
    def __init__(
        self,
        data: Data,
        wl_config: Optional[WLConfig] = None
    ):
        """Initialize the GraphTreeProducer.
        
        Args:
            data: PyG Data containing:
                - adj: torch.Tensor or SparseTensor of shape [N, N]
                - edge_index: optional, [2, E]
                - x: node features [N, F]
            wl_config: Optional configuration for WL computation
        """
        self.data = data
        self.wl_config = wl_config or WLConfig()
        
        # Convert adjacency matrix to scipy csr format
        if hasattr(data, 'adj') and isinstance(data.adj, torch.Tensor):
            A = data.adj.cpu().numpy()
            self.adj = sp.csr_matrix(A)
        elif hasattr(data, 'adj') and isinstance(data.adj, SparseTensor):
            coo = data.adj.to_scipy()
            self.adj = coo.tocsr()
        else:
            edge = data.edge_index.cpu().numpy()
            N = data.num_nodes
            coo = sp.coo_matrix((np.ones(edge.shape[1]), (edge[0], edge[1])), shape=(N, N))
            self.adj = coo.tocsr()
            
        self.N = self.adj.shape[0]
        self.features = data.x  # torch.Tensor [N, F]
        self.neighbors: Dict[int, set] = {}
        self._processed_adj: Optional[sp.csr_matrix] = None
        self._wl_reprs: Optional[torch.Tensor] = None

    def build_neighborhood_dict(self) -> Dict[int, set]:
        """Build 1-hop neighborhood for each node (including self).
        
        Returns:
            Dictionary mapping node indices to their neighborhood sets
        """
        nbrs = {}
        for node in range(self.N):
            row = self.adj[node]
            cols = row.tocoo().col
            s = set(cols.tolist())
            if self.wl_config.add_self_loops:
                s.add(node)
            nbrs[node] = s
        self.neighbors = nbrs
        return nbrs

    def _process_adjacency(self) -> sp.csr_matrix:
        """Process the adjacency matrix for WL computation.
        
        Returns:
            Processed adjacency matrix in CSR format
        """
        if self._processed_adj is not None:
            return self._processed_adj
            
        A = self.adj.tolil()
        
        if self.wl_config.add_self_loops:
            A.setdiag(1)
            
        if self.wl_config.symmetric:
            A = A + A.T
            A[A > 1] = 1
            
        if self.wl_config.normalize_adj:
            # Normalize adjacency matrix
            rowsum = np.array(A.sum(1)).flatten()
            mask = rowsum == 0
            rowsum[mask] = 1
            inv = 1.0 / rowsum
            D_inv = sp.diags(inv)
            A = D_inv.dot(A).dot(D_inv)
            
        self._processed_adj = A.tocsr()
        return self._processed_adj

    def compute_wl_representations(self) -> torch.Tensor:
        """Compute WL representations through multiple iterations.
        
        Returns:
            Tensor of shape [N, F] containing the final WL representations
        """
        A = self._process_adjacency()
        A = A.tocoo().astype(np.float32)
        
        # Convert to SparseTensor using the newer method
        indices = torch.stack([
            torch.LongTensor(A.row),
            torch.LongTensor(A.col)
        ])
        values = torch.FloatTensor(A.data)
        size = torch.Size(A.shape)
        
        # Create sparse tensor
        adj_t = torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=size,
            dtype=torch.float32
        )
        
        # Convert to SparseTensor for torch_sparse operations
        adj_sparse = SparseTensor(
            row=adj_t._indices()[0],
            col=adj_t._indices()[1],
            value=adj_t._values(),
            sparse_sizes=adj_t.size()
        )
        
        # Multiple WL iterations
        current_features = self.features
        for _ in range(self.wl_config.iterations):
            current_features = sparse_matmul(adj_sparse, current_features)
            
        self._wl_reprs = current_features
        return current_features

    def compute_pairwise_distances(
        self,
        wl_reprs: Optional[torch.Tensor] = None,
        frac_to_sample: float = 1.0,
        metric: Literal["l2", "cosine"] = "l2"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute pairwise distances among WL representations.
        
        Args:
            wl_reprs: Optional WL representations. If None, will compute them.
            frac_to_sample: Fraction of nodes to sample for distance computation
            metric: Distance metric to use ("l2" or "cosine")
            
        Returns:
            Tuple of (distance matrix, sampled indices)
        """
        if wl_reprs is None:
            wl_reprs = self.compute_wl_representations()
            
        X = wl_reprs.cpu().numpy()
        n = X.shape[0]
        
        if frac_to_sample >= 1.0:
            idx = np.arange(n)
            if metric == "l2":
                dist = np.linalg.norm(X[:, None] - X[None, :], axis=2)
            else:  # cosine
                X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
                dist = 1 - X_norm @ X_norm.T
        else:
            m = max(int(frac_to_sample * n), 1)
            idx = np.random.choice(n, size=m, replace=False)
            if metric == "l2":
                dist = np.linalg.norm(X[idx, None] - X[None, :], axis=2)
            else:  # cosine
                X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
                dist = 1 - X_norm[idx] @ X_norm.T
                
        return dist, idx

    def get_node_embedding(self, node: int) -> torch.Tensor:
        """Get the WL representation for a specific node.
        
        Args:
            node: Node index
            
        Returns:
            Tensor containing the node's WL representation
        """
        if self._wl_reprs is None:
            self.compute_wl_representations()
        return self._wl_reprs[node]

    def get_neighborhood_embedding(self, node: int) -> torch.Tensor:
        """Get the average WL representation of a node's neighborhood.
        
        Args:
            node: Node index
            
        Returns:
            Tensor containing the average neighborhood embedding
        """
        if not self.neighbors:
            self.build_neighborhood_dict()
            
        if self._wl_reprs is None:
            self.compute_wl_representations()
            
        neighbors = self.neighbors[node]
        return torch.mean(self._wl_reprs[list(neighbors)], dim=0)

    def compute_reverse_knn(
        self,
        distances: np.ndarray,
        sampled_indices: np.ndarray,
        k: int
    ) -> Dict[str, Union[np.ndarray, Dict[int, set]]]:
        """Compute k-NN and reverse k-NN based on WL distances.
        
        Args:
            distances: Distance matrix of shape (m, n) where m is number of sampled nodes
            sampled_indices: Array of sampled node indices
            k: Number of nearest neighbors to consider
            
        Returns:
            Dictionary containing:
            - "knn": Array of shape (m, k) containing k-NN indices for each sampled node
            - "rknn": Dictionary mapping original node indices to sets of sampled node indices
        """
        m, n = distances.shape
        # Ensure k doesn't exceed the number of nodes
        if k > n:
            import warnings
            warnings.warn(f"Requested k={k} is larger than number of nodes {n}. Reducing k to {n}.")
            k = n
        
        # Compute k-NN for each sampled node using argsort to ensure exactly k neighbors
        knn = np.array([
            np.argsort(distances[i])[:k] 
            for i in tqdm(range(m), desc="Computing k-NN")
        ])
        
        # Compute reverse k-NN
        rknn = defaultdict(set)
        for i, nbrs in enumerate(knn):
            for j in nbrs:
                rknn[int(j)].add(int(sampled_indices[i]))
                
        return {"knn": knn, "rknn": dict(rknn)}

    def select_exemplars(
        self,
        rknn_dict: Dict[int, set]
    ) -> List[int]:
        """Select exemplar nodes using CELF-accelerated greedy algorithm.
        
        Args:
            rknn_dict: Dictionary mapping node indices to sets of sampled node indices
            
        Returns:
            List of selected exemplar node indices
        """
        covered = set()
        selected = []
        
        # Initialize priority queue with (-coverage_size, node_idx) pairs
        pq = [(-len(nei), idx) for idx, nei in rknn_dict.items()]
        heapq.heapify(pq)
        
        # Select first node with maximum coverage
        neg_gain, node = heapq.heappop(pq)
        covered |= rknn_dict[node]
        selected.append(node)
        
        # Iteratively select remaining nodes
        while pq:
            neg_gain, node = heapq.heappop(pq)
            
            # If current max gain is 0, select all remaining nodes
            if neg_gain == 0:
                selected.extend(idx for _, idx in pq)
                selected.append(node)
                break
                
            # Recompute coverage gain for current node
            gain = len(rknn_dict[node] - covered)
            max_node, max_gain = node, gain
            
            # Update stale nodes in queue
            if pq and -pq[0][0] > gain:
                temp = [(gain, node)]
                while pq and -pq[0][0] > gain:
                    stale_neg, stale_node = heapq.heappop(pq)
                    new_gain = len(rknn_dict[stale_node] - covered)
                    temp.append((new_gain, stale_node))
                    if new_gain > max_gain:
                        max_gain, max_node = new_gain, stale_node
                        
                # Push updated nodes back to queue
                for g, n in temp:
                    if n != max_node:
                        heapq.heappush(pq, (-g, n))
                        
            # Select node with maximum gain
            selected.append(max_node)
            covered |= rknn_dict[max_node]
            rknn_dict.pop(max_node, None)
            
        return selected

    def sample_and_select_exemplars(
        self,
        k: int = 10,
        frac_to_sample: float = 0.05,
        metric: Literal["l2", "cosine"] = "cosine"
    ) -> List[int]:
        """Convenience method to perform sampling, compute reverse k-NN, and select exemplars.
        
        Args:
            k: Number of nearest neighbors to consider
            frac_to_sample: Fraction of nodes to sample
            metric: Distance metric to use
            
        Returns:
            List of selected exemplar node indices
        """
        # Compute distances and sample nodes
        distances, sampled_indices = self.compute_pairwise_distances(
            frac_to_sample=frac_to_sample,
            metric=metric
        )
        
        # Compute reverse k-NN
        rknn_result = self.compute_reverse_knn(
            distances=distances,
            sampled_indices=sampled_indices,
            k=k
        )
        
        # Select exemplars
        exemplars = self.select_exemplars(rknn_result["rknn"])
        
        return exemplars

if __name__ == "__main__":
    import torch
    from torch_geometric.data import Data

    # Create a small graph with 5 nodes
    # Node features: 3-dimensional vectors
    node_features = torch.tensor([
        [1.0, 0.0, 0.0],  # Node 0
        [0.0, 1.0, 0.0],  # Node 1
        [0.0, 0.0, 1.0],  # Node 2
        [1.0, 1.0, 0.0],  # Node 3
        [0.0, 1.0, 1.0],  # Node 4
    ], dtype=torch.float32)

    # Create adjacency matrix (undirected graph)
    adj_matrix = torch.tensor([
        [0, 1, 0, 1, 0],  # Node 0 connects to 1 and 3
        [1, 0, 1, 0, 0],  # Node 1 connects to 0 and 2
        [0, 1, 0, 0, 1],  # Node 2 connects to 1 and 4
        [1, 0, 0, 0, 1],  # Node 3 connects to 0 and 4
        [0, 0, 1, 1, 0],  # Node 4 connects to 2 and 3
    ], dtype=torch.float32)

    # Create PyG Data object
    data = Data(
        x=node_features,
        adj=adj_matrix,
        num_nodes=5
    )

    # Initialize GraphTreeProducer with custom config
    config = WLConfig(
        add_self_loops=True,
        normalize_adj=True,
        iterations=2,
        symmetric=True
    )
    producer = GraphTreeProducer(data, wl_config=config)

    # Build neighborhood dictionary
    neighbors = producer.build_neighborhood_dict()
    print("Neighborhoods:")
    for node, nbrs in neighbors.items():
        print(f"Node {node}: {nbrs}")

    # Compute WL representations
    wl_reprs = producer.compute_wl_representations()
    print("\nWL Representations shape:", wl_reprs.shape)
    print("WL Representations:", wl_reprs)

    # Compute pairwise distances
    distances, indices = producer.compute_pairwise_distances(
        wl_reprs=wl_reprs,
        frac_to_sample=1.0,  # Use all nodes
        metric="cosine"
    )
    print("\nDistance matrix shape:", distances.shape)
    print("First row of distances:", distances[0])

    # Get node-specific embeddings
    node_0_embedding = producer.get_node_embedding(0)
    print("\nNode 0 embedding:", node_0_embedding)

    # Get neighborhood embedding
    neighborhood_embedding = producer.get_neighborhood_embedding(0)
    print("\nNeighborhood embedding for node 0:", neighborhood_embedding)

    # # Option 1: Use the convenience method
    # exemplars = producer.sample_and_select_exemplars(
    #     k=10,
    #     frac_to_sample=0.05,
    #     metric="cosine"
    # )

    # Option 2: Use individual steps
    distances, sampled_indices = producer.compute_pairwise_distances(
        frac_to_sample=0.05,
        metric="cosine"
    )
    rknn_result = producer.compute_reverse_knn(
        distances=distances,
        sampled_indices=sampled_indices,
        k=3
    )
    exemplars = producer.select_exemplars(rknn_result["rknn"])
    print("\nSelected exemplars:", exemplars)