"""Readable CPU graph generators and measurements for FFR120, Chapter 12.

All generators produce simple, undirected, unweighted NetworkX graphs.
The optional ``rng`` is for explicit reproducibility experiments only:
normal teaching runs draw fresh random realizations.
"""

from __future__ import annotations

import networkx as nx
import numpy as np


def adjacency_to_edges(A):
    """Create a list of edges from a symmetric adjacency matrix."""

    i, j = np.where(np.triu(A, k=1))
    edges = list(zip(i, j))

    return edges


def edges_to_adjacency(edges, n=None):
    """Create a symmetric adjacency matrix from a list of edges."""

    if n is None:
        n = max(max(u, v) for u, v in edges) + 1

    A = np.zeros((n, n), dtype=int)

    if edges:
        i, j = np.array(edges).T
        A[i, j] = 1
        A[j, i] = 1

    return A


def erdos_renyi(n: int, p: float, rng=None) -> nx.Graph:
    """Independently include each of n(n - 1)/2 possible edges with probability p."""
    
    if n < 1 or not 0 <= p <= 1:
        raise ValueError("Require n >= 1 and 0 <= p <= 1.")
    
    rng = np.random.default_rng() if rng is None else rng
    
    graph = nx.empty_graph(n)
    
    row, col = np.triu_indices(n, k=1)
    
    selected = rng.random(len(row)) < p
    
    graph.add_edges_from(zip(row[selected].tolist(), col[selected].tolist()))
    
    return graph


def watts_strogatz(n: int, c: int, p: float, rng=None) -> nx.Graph:
    """Rewire each clockwise ring edge with probability p, avoiding duplicates.

    This implements the standard Watts-Strogatz rewiring operation: the source
    node remains fixed, while the other endpoint is replaced. Graphs need not
    remain connected. ``c`` must be even and smaller than ``n``.
    """
    if n < 3 or c < 2 or c >= n or c % 2 or not 0 <= p <= 1:
        raise ValueError("Require n >= 3, even 2 <= c < n, and 0 <= p <= 1.")
    
    rng = np.random.default_rng() if rng is None else rng
    graph = nx.empty_graph(n)
    
    for source in range(n):
        for offset in range(1, c // 2 + 1):
            graph.add_edge(source, (source + offset) % n)

    # Visit each original clockwise edge once (not newly created edges).
    for source in range(n):
        for offset in range(1, c // 2 + 1):
            target = (source + offset) % n
            if rng.random() >= p or graph.degree(source) == n - 1:
                continue
            graph.remove_edge(source, target)
            candidates = [j for j in range(n)
                          if j != source and not graph.has_edge(source, j)]
            new_target = int(rng.choice(candidates))
            graph.add_edge(source, new_target)
    
    return graph


def preferential_growth(
    n: int, m: int, n0: int | None = None, rng=None
) -> nx.Graph:
    """Grow from a complete seed, attaching each new node to m distinct nodes.

    Targets are sampled simultaneously without replacement, weighted by degrees
    measured *before* the new node is added. For the pure BA asymptotics, the
    difference between a complete finite seed and the minimal seed vanishes.
    """
    n0 = m + 1 if n0 is None else n0
    if m < 1 or n0 < max(2, m) or n < n0:
        raise ValueError("Require m >= 1, n0 >= max(2, m), n >= n0.")
    rng = np.random.default_rng() if rng is None else rng
    graph = nx.complete_graph(n0)
    for new_node in range(n0, n):
        degrees = np.fromiter((graph.degree(j) for j in range(new_node)),
                              dtype=float, count=new_node)
        probabilities = degrees / degrees.sum()
        targets = rng.choice(new_node, size=m, replace=False, p=probabilities)
        graph.add_node(new_node)
        graph.add_edges_from((new_node, int(target)) for target in targets)
    return graph


def graph_summary(graph: nx.Graph) -> dict:
    """Measure size, density, clustering coefficient, and path metrics on its largest component.

    A disconnected network has no finite all-pairs distance. We explicitly
    report the largest-connected-component (LCC) fraction and measure distances
    inside that component, rather than silently discarding disconnected nodes.
    """
    if graph.number_of_nodes() == 0:
        raise ValueError("Graph must contain at least one node.")
        
    components = list(nx.connected_components(graph))
    
    largest = graph.subgraph(max(components, key=len))
    
    return {
        "n": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "mean_degree": 2 * graph.number_of_edges() / graph.number_of_nodes(),
        "C": nx.transitivity(graph),
        "mean_path_lcc": nx.average_shortest_path_length(largest),
        "diameter_lcc": nx.diameter(largest),
    }


def graph_full_summary(graph: nx.Graph) -> dict:
    """Measure size, density, clustering coefficient, and path metrics on its largest component.

    A disconnected network has no finite all-pairs distance. We explicitly
    report the largest-connected-component (LCC) fraction and measure distances
    inside that component, rather than silently discarding disconnected nodes.
    """
    if graph.number_of_nodes() == 0:
        raise ValueError("Graph must contain at least one node.")
    components = list(nx.connected_components(graph))
    largest = graph.subgraph(max(components, key=len))
    return {
        "n": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "mean_degree": 2 * graph.number_of_edges() / graph.number_of_nodes(),
        "C": nx.transitivity(graph),
        "mean_local_clustering": nx.average_clustering(graph),
        "lcc_fraction": largest.number_of_nodes() / graph.number_of_nodes(),
        "mean_path_lcc": nx.average_shortest_path_length(largest),
        "diameter_lcc": nx.diameter(largest),
    }
