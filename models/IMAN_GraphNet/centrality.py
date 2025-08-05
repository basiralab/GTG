import numpy as np
import networkx as nx


# put it back into a 2D symmetric array


def topological_measures(data):
    # ROI is the number of brain regions (i.e.,35 in our case)
    ROI = 160

    topology = []



    # A = to_2d(data)
    np.fill_diagonal(data, 0)

    # create a graph from similarity matrix
    G = nx.from_numpy_matrix(np.absolute(data))
    U = G.to_undirected()

    # Centrality #

    # compute closeness centrality and transform the output to vector
    cc = nx.closeness_centrality(U, distance="weight")
    closeness_centrality = np.array([cc[g] for g in U])
    # compute betweeness centrality and transform the output to vector
    # bc = nx.betweenness_centrality(U, weight='weight')
    # bc = (nx.betweenness_centrality(U))
    betweenness_centrality = np.array([cc[g] for g in U])
    # # compute egeinvector centrality and transform the output to vector
    ec = nx.eigenvector_centrality_numpy(U)
    eigenvector_centrality = np.array([ec[g] for g in U])


    topology.append(closeness_centrality)  # 0
    topology.append(betweenness_centrality)  # 1
    topology.append(eigenvector_centrality)  # 2

    return topology
# put it back into a 2D symmetric array

def eigen_centrality(data):
    """Calculate eigenvector centrality for a graph"""
    # Convert numpy array to graph using newer API
    G = nx.from_numpy_array(np.absolute(data))
    
    # Calculate eigenvector centrality
    centrality = nx.eigenvector_centrality(G, max_iter=1000)
    
    # Convert to numpy array
    n = len(centrality)
    eigenvector = np.zeros(n)
    for i in range(n):
        eigenvector[i] = centrality[i]
    
    return eigenvector

