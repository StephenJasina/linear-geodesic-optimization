import typing

import networkx as nx
import numpy as np
import ot


# TODO: Add back older computation methods

def ricci_curvature_optimal_transport(
    graph: nx.Graph,
    edge_distance_label: typing.Optional[str] = None,
    edge_weight_label: typing.Optional[str] = None,
    alpha: float = 0.9999
) -> typing.Dict[typing.Tuple[int, int], float]:
    # Create mappings between numerical IDs and node names
    index_to_node = [node for node in graph.nodes]
    node_to_index = {node: index for index, node in enumerate(index_to_node)}
    n_nodes = len(index_to_node)

    # Relabel the graph
    graph = nx.relabel_nodes(graph, node_to_index)

    # Create an augmented graph. Essentially, add a duplicate edge for
    # each node incident to a particular node.
    # Simultaneously, create dictionaries of the edge weights.
    weights = []
    index_to_node_augmented = list(range(n_nodes))
    for index in range(n_nodes):
        neighbors = list(graph.neighbors(index))
        weights_index = {}
        for neighbor in neighbors:
            graph.add_edge(
                index, (index, neighbor),
                weight = 1. if edge_weight_label is None else graph.edges[index, neighbor][edge_weight_label]
            )
            weights_index[neighbor] = (
                1. if edge_weight_label is None
                else graph.edges[index, neighbor][edge_weight_label]
            )
            index_to_node_augmented.append((index, neighbor))
        weights.append(weights_index)
    n_nodes_augmented = len(index_to_node_augmented)
    node_to_index_augmented = {node: index for index, node in enumerate(index_to_node_augmented)}

    # Compute the distance matrix, which is just a matrix of shortest
    # paths
    distance_matrix = np.zeros((n_nodes_augmented, n_nodes_augmented))
    if edge_distance_label is None:
        distances_iterator = nx.all_pairs_shortest_path_length(graph)
    else:
        distances_iterator = nx.all_pairs_dijkstra_path_length(
            graph, weight = edge_distance_label
        )
    for source, distance_dict in distances_iterator:
        index_source = node_to_index_augmented[source]
        for destination, distance in distance_dict.items():
            index_destination = node_to_index_augmented[destination]
            distance_matrix[index_source, index_destination] = distance

    ricci_curvatures = {}
    for source in range(n_nodes):
        weights_source = weights[source]
        for destination in weights_source:
            # Don't do duplicate work
            if source > destination:
                continue

            # TODO: Should these distributions depend on distances?
            distribution_source = np.zeros(n_nodes_augmented)
            for neighbor, neighbor_weight in weights_source.items():
                if neighbor == destination:
                    distribution_source[node_to_index_augmented[source, neighbor]] = weights_source[destination]
                else:
                    distribution_source[neighbor] = weights_source[neighbor]
            distribution_source_sum = np.sum(distribution_source)
            if distribution_source_sum != 0.:
                distribution_source *= (1. - alpha) / distribution_source_sum
                distribution_source[source] = alpha
            else:
                distribution_source[source] = 1.

            distribution_destination = np.zeros(n_nodes_augmented)
            weights_destination = weights[destination]
            for neighbor, neighbor_weight in weights[destination].items():
                if neighbor == source:
                    distribution_destination[node_to_index_augmented[destination, neighbor]] = weights_destination[source]
                else:
                    distribution_destination[neighbor] = weights_destination[neighbor]
            distribution_destination_sum = np.sum(distribution_destination)
            if distribution_destination_sum != 0.:
                distribution_destination *= (1. - alpha) / distribution_destination_sum
                distribution_destination[destination] = alpha
            else:
                distribution_destination[destination] = 1.

            transport_distance = ot.emd2(distribution_source, distribution_destination, distance_matrix)
            edge_distance = 1. if edge_distance_label is None else graph.edges[source, destination][edge_distance_label]
            ricci_curvatures[index_to_node[source], index_to_node[destination]] = (1. - transport_distance / edge_distance) / (1. - alpha)

    return ricci_curvatures
