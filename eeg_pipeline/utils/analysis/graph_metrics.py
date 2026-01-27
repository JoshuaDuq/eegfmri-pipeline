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

from typing import Dict, Optional, Tuple

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
    
    Raises
    ------
    ValueError
        If adjacency matrix is not square
    """
    adjacency = np.asarray(adj, dtype=float)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(
            f"Adjacency must be square; got shape {adjacency.shape}"
        )
    adjacency = 0.5 * (adjacency + adjacency.T)
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


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
    
    Raises
    ------
    ValueError
        If top_proportion is not in [0, 1]
    """
    adjacency = np.asarray(adj, dtype=float)
    
    if top_proportion is not None:
        if not 0.0 <= top_proportion <= 1.0:
            raise ValueError(
                f"top_proportion must be in [0, 1]; got {top_proportion}"
            )
        upper_triangle_indices = np.triu_indices_from(adjacency, k=1)
        edge_values = adjacency[upper_triangle_indices]
        finite_values = edge_values[np.isfinite(edge_values)]
        
        if len(finite_values) == 0:
            return np.zeros_like(adjacency)
        
        percentile_threshold = 100 * (1 - top_proportion)
        threshold = np.percentile(finite_values, percentile_threshold)
    
    if threshold is None:
        threshold = 0.0
    
    binary_adjacency = (adjacency >= threshold).astype(float)
    np.fill_diagonal(binary_adjacency, 0.0)
    return binary_adjacency


# =============================================================================
# Graph Metrics
# =============================================================================

def compute_global_efficiency_weighted(
    adj: np.ndarray, eps: float = 1e-9
) -> float:
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
        Global efficiency value, or np.nan if computation fails
    """
    graph = nx.from_numpy_array(adj)
    
    edge_lengths = {}
    for source, target, edge_data in graph.edges(data=True):
        weight = abs(edge_data.get("weight", 0.0))
        edge_lengths[(source, target)] = 1.0 / (weight + eps)
    nx.set_edge_attributes(graph, edge_lengths, "length")

    try:
        shortest_path_lengths = dict(
            nx.all_pairs_dijkstra_path_length(graph, weight="length")
        )
    except (nx.NetworkXError, ValueError, KeyError):
        return np.nan

    num_nodes = graph.number_of_nodes()
    if num_nodes <= 1:
        return np.nan

    inverse_distances = []
    for source_node in range(num_nodes):
        for target_node in range(source_node + 1, num_nodes):
            path_length = shortest_path_lengths.get(source_node, {}).get(
                target_node, np.inf
            )
            if np.isfinite(path_length) and path_length > 0:
                inverse_distances.append(1.0 / path_length)

    if not inverse_distances:
        return np.nan

    normalization_factor = 2.0 / (num_nodes * (num_nodes - 1))
    return float(normalization_factor * np.sum(inverse_distances))


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
        Small-world sigma value, or np.nan if computation fails
    """
    if n_rand < 1:
        raise ValueError(f"n_rand must be >= 1; got {n_rand}")
    
    graph = nx.from_numpy_array(adj_bin)
    
    if nx.number_of_nodes(graph) < 3 or nx.number_of_edges(graph) == 0:
        return np.nan
    
    if not nx.is_connected(graph):
        largest_component = max(nx.connected_components(graph), key=len)
        graph = graph.subgraph(largest_component).copy()
    
    try:
        return float(nx.sigma(graph, niter=n_rand, seed=42))
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
    num_nodes = adj.shape[0]
    
    if not community_labels:
        return np.full(num_nodes, np.nan, dtype=float)
    
    unique_communities = sorted(set(community_labels.values()))
    if len(unique_communities) < 2:
        return np.full(num_nodes, np.nan, dtype=float)
    
    adjacency = np.maximum(adj, 0.0)
    node_degrees = adjacency.sum(axis=1)
    participation_coefficients = np.full(num_nodes, np.nan, dtype=float)
    
    for node_idx in range(num_nodes):
        node_degree = node_degrees[node_idx]
        if node_degree <= 0:
            continue
        
        squared_proportion_sum = 0.0
        for community in unique_communities:
            community_node_indices = [
                j
                for j, label in community_labels.items()
                if label == community
            ]
            if not community_node_indices:
                continue
            connections_to_community = np.sum(
                adjacency[node_idx, community_node_indices]
            )
            proportion = connections_to_community / node_degree
            squared_proportion_sum += proportion ** 2
        
        participation_coefficients[node_idx] = 1.0 - squared_proportion_sum
    
    return participation_coefficients



