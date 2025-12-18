"""
Graph Metrics Utilities
========================

Network/graph theory metrics for connectivity analysis.
These functions operate on adjacency matrices (numpy arrays).

Metrics:
- Global efficiency (weighted)
- Small-world sigma
- Participation coefficient
- Clustering coefficient
- Thresholding utilities
"""

from __future__ import annotations

from typing import Dict, Optional, Any

import numpy as np
import networkx as nx


# =============================================================================
# Matrix Utilities
# =============================================================================

def symmetrize_adjacency(adj: np.ndarray) -> np.ndarray:
    """
    Symmetrize and clean an adjacency matrix.
    
    Parameters
    ----------
    adj : np.ndarray
        Square adjacency matrix
    
    Returns
    -------
    np.ndarray
        Symmetrized matrix with zero diagonal
    """
    adj = np.asarray(adj, dtype=float)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"Adjacency must be square; got shape {adj.shape}")
    adj = 0.5 * (adj + adj.T)
    np.fill_diagonal(adj, 0.0)
    return adj


def threshold_adjacency(
    adj: np.ndarray,
    threshold: Optional[float] = None,
    top_proportion: Optional[float] = None,
) -> np.ndarray:
    """
    Threshold an adjacency matrix.
    
    Parameters
    ----------
    adj : np.ndarray
        Adjacency matrix
    threshold : float, optional
        Absolute threshold value
    top_proportion : float, optional
        Keep top proportion of edges (0-1)
    
    Returns
    -------
    np.ndarray
        Binary adjacency matrix
    """
    adj = np.asarray(adj, dtype=float)
    
    if top_proportion is not None:
        # Get upper triangle values
        triu_idx = np.triu_indices_from(adj, k=1)
        values = adj[triu_idx]
        values = values[np.isfinite(values)]
        
        if len(values) == 0:
            return np.zeros_like(adj)
        
        threshold = np.percentile(values, 100 * (1 - top_proportion))
    
    if threshold is None:
        threshold = 0.0
    
    binary = (adj >= threshold).astype(float)
    np.fill_diagonal(binary, 0.0)
    return binary


# =============================================================================
# Graph Metrics
# =============================================================================

def compute_global_efficiency_weighted(adj: np.ndarray, eps: float = 1e-9) -> float:
    """
    Compute weighted global efficiency of a network.
    
    Global efficiency is the average inverse shortest path length,
    measuring how efficiently information can be exchanged.
    
    Parameters
    ----------
    adj : np.ndarray
        Weighted adjacency matrix
    eps : float
        Small value to avoid division by zero
    
    Returns
    -------
    float
        Global efficiency value
    """
    G = nx.from_numpy_array(adj)
    
    # Convert weights to lengths (higher weight = shorter path)
    lengths = {}
    for u, v, data in G.edges(data=True):
        w = abs(data.get("weight", 0.0))
        lengths[(u, v)] = 1.0 / (w + eps)
    nx.set_edge_attributes(G, lengths, "length")

    try:
        sp_lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight="length"))
    except (nx.NetworkXError, ValueError, KeyError):
        return np.nan

    n = G.number_of_nodes()
    if n <= 1:
        return np.nan

    inv_dist = []
    for i in range(n):
        for j in range(i + 1, n):
            d = sp_lengths.get(i, {}).get(j, np.inf)
            if np.isfinite(d) and d > 0:
                inv_dist.append(1.0 / d)

    if not inv_dist:
        return np.nan

    return float((2.0 / (n * (n - 1))) * np.sum(inv_dist))


def compute_small_world_sigma(
    adj_bin: np.ndarray,
    n_rand: int = 100,
) -> float:
    """
    Compute small-world sigma coefficient.
    
    Sigma = (C/C_rand) / (L/L_rand)
    where C is clustering and L is path length.
    Sigma > 1 indicates small-world topology.
    
    Parameters
    ----------
    adj_bin : np.ndarray
        Binary adjacency matrix
    n_rand : int
        Number of random networks for comparison
    
    Returns
    -------
    float
        Small-world sigma value
    """
    G = nx.from_numpy_array(adj_bin)
    
    if nx.number_of_nodes(G) < 3 or nx.number_of_edges(G) == 0:
        return np.nan
    
    # Use largest connected component
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    
    try:
        return float(nx.sigma(G, niter=n_rand, seed=42))
    except (nx.NetworkXError, ZeroDivisionError, ValueError):
        return np.nan


def compute_participation_coefficient(
    adj: np.ndarray,
    community_labels: Dict[int, str],
) -> np.ndarray:
    """
    Compute participation coefficient for each node.
    
    Participation coefficient measures how evenly a node's connections
    are distributed across communities. High P = diverse connections.
    
    Parameters
    ----------
    adj : np.ndarray
        Adjacency matrix
    community_labels : Dict[int, str]
        Mapping from node index to community label
    
    Returns
    -------
    np.ndarray
        Participation coefficient for each node
    """
    n = adj.shape[0]
    
    if not community_labels:
        return np.full(n, np.nan, dtype=float)
    
    unique_comms = sorted(set(community_labels.values()))
    if len(unique_comms) < 2:
        return np.full(n, np.nan, dtype=float)
    
    adj = np.maximum(adj, 0.0)
    deg = adj.sum(axis=1)
    pc = np.full(n, np.nan, dtype=float)
    
    for i in range(n):
        k_i = deg[i]
        if k_i <= 0:
            continue
        
        accum = 0.0
        for comm in unique_comms:
            idx = [j for j, c in community_labels.items() if c == comm]
            if not idx:
                continue
            k_ic = np.sum(adj[i, idx])
            accum += (k_ic / k_i) ** 2
        
        pc[i] = 1.0 - accum
    
    return pc


def compute_clustering_coefficient(adj: np.ndarray, weighted: bool = True) -> np.ndarray:
    """Compute clustering coefficient per node."""
    G = nx.from_numpy_array(adj)
    weight = "weight" if weighted else None
    cc_dict = nx.clustering(G, weight=weight)
    return np.array([cc_dict.get(i, np.nan) for i in range(adj.shape[0])])


def compute_betweenness_centrality(adj: np.ndarray) -> np.ndarray:
    """Compute betweenness centrality per node."""
    G = nx.from_numpy_array(adj)
    if G.number_of_edges() == 0:
        return np.full(adj.shape[0], np.nan)
    try:
        bc = nx.betweenness_centrality(G, weight="weight")
        return np.array([bc.get(i, np.nan) for i in range(adj.shape[0])])
    except (nx.NetworkXError, ValueError, ZeroDivisionError):
        return np.full(adj.shape[0], np.nan)


def compute_eigenvector_centrality(adj: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """Compute eigenvector centrality per node."""
    G = nx.from_numpy_array(adj)
    if G.number_of_edges() == 0:
        return np.full(adj.shape[0], np.nan)
    try:
        ec = nx.eigenvector_centrality_numpy(G, weight="weight")
        return np.array([ec.get(i, np.nan) for i in range(adj.shape[0])])
    except (nx.NetworkXError, ValueError, np.linalg.LinAlgError):
        return np.full(adj.shape[0], np.nan)


def compute_rich_club_coefficient(adj: np.ndarray, k: int = None) -> float:
    """Compute rich club coefficient (tendency of high-degree nodes to connect)."""
    G = nx.from_numpy_array(adj > 0)  # Binary graph
    if G.number_of_edges() == 0:
        return np.nan
    try:
        rc = nx.rich_club_coefficient(G, normalized=False)
        if not rc:
            return np.nan
        if k is not None and k in rc:
            return float(rc[k])
        degrees = [d for _, d in G.degree()]
        if not degrees:
            return np.nan
        median_k = int(np.median(degrees))
        return float(rc.get(median_k, np.nan))
    except (nx.NetworkXError, ValueError, ZeroDivisionError, KeyError):
        return np.nan


def compute_characteristic_path_length(adj: np.ndarray) -> float:
    """Compute characteristic path length (average shortest path)."""
    G = nx.from_numpy_array(adj)
    if G.number_of_edges() == 0:
        return np.nan
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    if G.number_of_nodes() < 2:
        return np.nan
    try:
        return float(nx.average_shortest_path_length(G, weight=None))
    except (nx.NetworkXError, ValueError, ZeroDivisionError):
        return np.nan


def compute_network_segregation_integration(
    adj: np.ndarray, community_map: dict, labels: np.ndarray
) -> tuple:
    """Compute network segregation and integration. Returns (segregation, integration)."""
    from typing import Tuple
    if not community_map or adj.size == 0:
        return np.nan, np.nan
    comms = [community_map.get(str(l), None) for l in labels]
    n_nodes = adj.shape[0]
    within_sum = between_sum = total_sum = 0.0
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            w = abs(adj[i, j])
            if not np.isfinite(w):
                continue
            total_sum += w
            if comms[i] is not None and comms[j] is not None:
                if comms[i] == comms[j]:
                    within_sum += w
                else:
                    between_sum += w
    if total_sum == 0:
        return np.nan, np.nan
    return float(within_sum / total_sum), float(between_sum / total_sum)



