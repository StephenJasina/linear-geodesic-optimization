import typing

import networkx as nx
import numpy as np
import ot


def ricci_curvature_optimal_transport(
    graph: nx.Graph,
    edge_distance_label: typing.Optional[str] = None,
    edge_weight_label: typing.Optional[str] = None,
    alpha: float = 0.9999
) -> typing.Dict[typing.Tuple[typing.Any, typing.Any], float]:
    # Create mappings between node names (generally strings) and
    # integers, which act as indices
    nodes_to_indices = {}
    indices_to_nodes = []
    for node in graph.nodes:
        nodes_to_indices[node] = len(indices_to_nodes)
        indices_to_nodes.append(node)
    n_nodes = len(indices_to_nodes)

    # Compute the distance matrix, which is just a matrix of shortest
    # paths
    distance_matrix = np.zeros((n_nodes, n_nodes))
    if edge_distance_label is None:
        distances_iterator = nx.all_pairs_shortest_path_length(graph)
    else:
        distances_iterator = nx.all_pairs_dijkstra_path_length(
            graph, weight = edge_distance_label
        )
    for source, distance_dict in distances_iterator:
        index_source = nodes_to_indices[source]
        for destination, distance in distance_dict.items():
            index_destination = nodes_to_indices[destination]
            distance_matrix[index_source, index_destination] = distance

    # Compute the relative probability that a chain is in a state at a
    # given time. That is, for each node, assign a probability (scaled
    # by `alpha`) that a random walk stays at the node in one step.
    # TODO: Is this doing the right thing when the edges are weighted?
    node_weights = np.ones(n_nodes)

    # Compute the initial distributions
    distributions = []
    for node in indices_to_nodes:
        n_neighbors = graph.degree[node]
        distribution = np.zeros(n_nodes)

        node_index = nodes_to_indices[node]
        node_probability = alpha * node_weights[node_index]

        distribution_sum = 0.
        for neighbor in graph.neighbors(node):
            distribution_neighbor = (
                1.
                if edge_weight_label is None
                else graph.edges[node, neighbor][edge_weight_label]
            ) / (
                1.
                if edge_distance_label is None
                else graph.edges[node, neighbor][edge_distance_label]**2
            )
            distribution[nodes_to_indices[neighbor]] = distribution_neighbor
            distribution_sum += distribution_neighbor
        if distribution_sum == 0.:
            distribution[node_index] = 1.
        else:
            distribution *= (1. - node_probability) / np.sum(distribution)
            distribution[node_index] += node_probability

        distributions.append(distribution)

    # For each edge, compute the optimal transport cost between the
    # distributions corresponding to the edge's endpoints. From there,
    # compute the Ollivier-Ricci curvature.
    ricci_curvatures = {}
    for source, destination, data in graph.edges(data=True):
        transport_distance = ot.emd2(
            distributions[nodes_to_indices[source]],
            distributions[nodes_to_indices[destination]],
            distance_matrix
        )
        edge_distance = 1. if edge_distance_label is None else data[edge_distance_label]
        ricci_curvatures[source, destination] = (1. - transport_distance / edge_distance) / (1. - alpha)

    return ricci_curvatures