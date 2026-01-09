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


def compute_clustering_coefficient_weighted(adj: np.ndarray) -> np.ndarray:
    """
    Compute weighted clustering coefficient per node.
    
    Parameters
    ----------
    adj : np.ndarray
        Weighted adjacency matrix
    
    Returns
    -------
    np.ndarray
        Clustering coefficient for each node
    """
    graph = nx.from_numpy_array(adj)
    clustering_dict = nx.clustering(graph, weight="weight")
    num_nodes = adj.shape[0]
    return np.array(
        [clustering_dict.get(node_idx, np.nan) for node_idx in range(num_nodes)]
    )


def compute_clustering_coefficient_binary(adj: np.ndarray) -> np.ndarray:
    """
    Compute binary clustering coefficient per node.
    
    Parameters
    ----------
    adj : np.ndarray
        Adjacency matrix (treated as binary)
    
    Returns
    -------
    np.ndarray
        Clustering coefficient for each node
    """
    graph = nx.from_numpy_array(adj)
    clustering_dict = nx.clustering(graph, weight=None)
    num_nodes = adj.shape[0]
    return np.array(
        [clustering_dict.get(node_idx, np.nan) for node_idx in range(num_nodes)]
    )


def compute_clustering_coefficient(adj: np.ndarray, weighted: bool = True) -> np.ndarray:
    """
    Compute clustering coefficient per node.
    
    Parameters
    ----------
    adj : np.ndarray
        Adjacency matrix
    weighted : bool
        If True, use weighted clustering; if False, use binary
    
    Returns
    -------
    np.ndarray
        Clustering coefficient for each node
    
    Note
    ----
    This function is maintained for backward compatibility.
    Prefer using compute_clustering_coefficient_weighted or
    compute_clustering_coefficient_binary for clarity.
    """
    if weighted:
        return compute_clustering_coefficient_weighted(adj)
    return compute_clustering_coefficient_binary(adj)


def compute_betweenness_centrality(adj: np.ndarray) -> np.ndarray:
    """
    Compute betweenness centrality per node.
    
    Betweenness centrality measures the fraction of shortest paths
    that pass through each node.
    
    Parameters
    ----------
    adj : np.ndarray
        Weighted adjacency matrix
    
    Returns
    -------
    np.ndarray
        Betweenness centrality for each node
    """
    graph = nx.from_numpy_array(adj)
    if graph.number_of_edges() == 0:
        return np.full(adj.shape[0], np.nan)
    try:
        betweenness_dict = nx.betweenness_centrality(graph, weight="weight")
        num_nodes = adj.shape[0]
        return np.array(
            [
                betweenness_dict.get(node_idx, np.nan)
                for node_idx in range(num_nodes)
            ]
        )
    except (nx.NetworkXError, ValueError, ZeroDivisionError):
        return np.full(adj.shape[0], np.nan)


def compute_eigenvector_centrality(
    adj: np.ndarray, max_iter: int = 100
) -> np.ndarray:
    """
    Compute eigenvector centrality per node.
    
    Eigenvector centrality measures a node's importance based on
    the importance of its neighbors.
    
    Parameters
    ----------
    adj : np.ndarray
        Weighted adjacency matrix
    max_iter : int
        Maximum iterations (not used by numpy implementation)
    
    Returns
    -------
    np.ndarray
        Eigenvector centrality for each node
    """
    graph = nx.from_numpy_array(adj)
    if graph.number_of_edges() == 0:
        return np.full(adj.shape[0], np.nan)
    try:
        eigenvector_dict = nx.eigenvector_centrality_numpy(
            graph, weight="weight"
        )
        num_nodes = adj.shape[0]
        return np.array(
            [
                eigenvector_dict.get(node_idx, np.nan)
                for node_idx in range(num_nodes)
            ]
        )
    except (nx.NetworkXError, ValueError, np.linalg.LinAlgError):
        return np.full(adj.shape[0], np.nan)


def compute_rich_club_coefficient(adj: np.ndarray, k: Optional[int] = None) -> float:
    """
    Compute rich club coefficient.
    
    Measures the tendency of high-degree nodes to connect to each other.
    
    Parameters
    ----------
    adj : np.ndarray
        Adjacency matrix (converted to binary)
    k : int, optional
        Degree threshold. If None, uses median degree
    
    Returns
    -------
    float
        Rich club coefficient, or np.nan if computation fails
    """
    binary_adjacency = adj > 0
    graph = nx.from_numpy_array(binary_adjacency)
    if graph.number_of_edges() == 0:
        return np.nan
    try:
        rich_club_dict = nx.rich_club_coefficient(graph, normalized=False)
        if not rich_club_dict:
            return np.nan
        if k is not None and k in rich_club_dict:
            return float(rich_club_dict[k])
        node_degrees = [degree for _, degree in graph.degree()]
        if not node_degrees:
            return np.nan
        median_degree = int(np.median(node_degrees))
        return float(rich_club_dict.get(median_degree, np.nan))
    except (nx.NetworkXError, ValueError, ZeroDivisionError, KeyError):
        return np.nan


def compute_characteristic_path_length(adj: np.ndarray) -> float:
    """
    Compute characteristic path length (average shortest path).
    
    Parameters
    ----------
    adj : np.ndarray
        Adjacency matrix (treated as unweighted)
    
    Returns
    -------
    float
        Characteristic path length, or np.nan if computation fails
    """
    graph = nx.from_numpy_array(adj)
    if graph.number_of_edges() == 0:
        return np.nan
    if not nx.is_connected(graph):
        largest_component = max(nx.connected_components(graph), key=len)
        graph = graph.subgraph(largest_component).copy()
    if graph.number_of_nodes() < 2:
        return np.nan
    try:
        return float(nx.average_shortest_path_length(graph, weight=None))
    except (nx.NetworkXError, ValueError, ZeroDivisionError):
        return np.nan


def compute_network_segregation_integration(
    adj: np.ndarray, community_map: Dict[str, str], labels: np.ndarray
) -> Tuple[float, float]:
    """
    Compute network segregation and integration.
    
    Segregation: proportion of connections within communities.
    Integration: proportion of connections between communities.
    
    Parameters
    ----------
    adj : np.ndarray
        Adjacency matrix
    community_map : Dict[str, str]
        Mapping from label string to community identifier
    labels : np.ndarray
        Node labels corresponding to adjacency matrix rows/columns
    
    Returns
    -------
    Tuple[float, float]
        (segregation, integration), or (np.nan, np.nan) if computation fails
    """
    if not community_map or adj.size == 0:
        return np.nan, np.nan
    
    node_communities = [
        community_map.get(str(label), None) for label in labels
    ]
    num_nodes = adj.shape[0]
    within_community_sum = 0.0
    between_community_sum = 0.0
    total_connection_sum = 0.0
    
    for source_node in range(num_nodes):
        for target_node in range(source_node + 1, num_nodes):
            connection_weight = abs(adj[source_node, target_node])
            if not np.isfinite(connection_weight):
                continue
            
            total_connection_sum += connection_weight
            source_community = node_communities[source_node]
            target_community = node_communities[target_node]
            
            if source_community is not None and target_community is not None:
                if source_community == target_community:
                    within_community_sum += connection_weight
                else:
                    between_community_sum += connection_weight
    
    if total_connection_sum == 0:
        return np.nan, np.nan
    
    segregation = float(within_community_sum / total_connection_sum)
    integration = float(between_community_sum / total_connection_sum)
    return segregation, integration



