import typing

import networkx as nx
import numpy as np
import numpy.typing as npt
import ot

import linear_geodesic_optimization.graph.distance as graph_distance


def get_augmented_graph(
    graph: nx.Graph,
    node_to_index: typing.Dict[typing.Any, int]
) -> nx.Graph:
    """
    Attach a star to each vertex of a graph.

    For each u and for each neighbor v of u, add an additional vertex
    v' and an edge from u to v'. Return both the new graph and a mapping
    from node labels of the old graph to corresponding node labels of
    the augmented graph.

    The labels of the nodes in the augmented graph are either:
    * Integers, if the node correspond to nodes in the original graph
    * Pairs (i, j) if the node is newly added and is connected to node i
      in the "direction" if node j

    Attributes of the duplicated nodes and edges will be identical.
    """
    n_nodes = len(node_to_index)

    graph_augmented = nx.relabel_nodes(graph, node_to_index)

    for index in range(n_nodes):
        neighbors = list(graph_augmented.neighbors(index))
        for neighbor in neighbors:
            # Make a new node with copied attributes
            graph_augmented.add_node((index, neighbor))
            for key, value in graph_augmented.nodes[neighbor].items():
                graph_augmented.nodes[(index, neighbor)][key] = value

            # Make a new edge with copied attributes
            graph_augmented.add_edge(index, (index, neighbor))
            for key, value in graph_augmented.edges[index, neighbor].items():
                graph_augmented.edges[index, (index, neighbor)][key] = value

    return graph_augmented

def compute_augmented_distances(
    graph_augmented: nx.Graph,
    distance_matrix: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Compute the distance matrix for an augmented graph.

    As input, take an augmented graph and the distance matrix for the
    corresponding non-augmented graph.

    As output, return a new distance matrix that is describes a metric
    on the augmented graph. The new metric space will have the old one
    as a subspac

    TODO: It would be nice if the nodes duplicated from the original
    graph would behave nearly identically to the original nodes. That
    is, we would have
    * d(i, (k, l)) = d(i, l)
    * d((i, j), k) = d(j, k)
    * d((i, j), (k, l)) = d(j, l)
    when i, j, k, and l are distinct.
    """
    n_nodes_augmented = graph_augmented.number_of_nodes()
    distance_matrix_augmented = np.zeros((n_nodes_augmented, n_nodes_augmented))

    for u_index, u in enumerate(graph_augmented.nodes):
        if isinstance(u, int):
            for v_index, v in enumerate(graph_augmented.nodes):
                if isinstance(v, int):
                    # Non-augmented -> non-augmented case
                    distance_matrix_augmented[u_index, v_index] = distance_matrix[u, v]
                else:
                    # Non-augmented -> augmented case
                    if u == v[0]:
                        distance_matrix_augmented[u_index, v_index] = distance_matrix[u, v[1]]
                    else:
                        distance_matrix_augmented[u_index, v_index] = distance_matrix[u, v[0]] + distance_matrix[v[0], v[1]]
        else:
            for v_index, v in enumerate(graph_augmented.nodes):
                if isinstance(v, int):
                    # Augmented -> non-augmented case
                    if u[0] == v:
                        distance_matrix_augmented[u_index, v_index] = distance_matrix[u[1], v]
                    else:
                        distance_matrix_augmented[u_index, v_index] = distance_matrix[u[1], u[0]] + distance_matrix[u[0], v]
                else:
                    # Augmented -> augmented case
                    distance_matrix_augmented[u_index, v_index] = distance_matrix[u[1], u[0]] + distance_matrix[u[0], v[0]] + distance_matrix[v[0], v[1]]

    return distance_matrix_augmented

def get_distribution(
    graph: nx.Graph,
    node_to_index: typing.Dict[typing.Any, int],
    x: int, alpha: float,
    edge_weight_label: typing.Optional[str] = None
) -> npt.NDArray[np.float64]:
    distribution = np.zeros(len(graph.nodes))

    for neighbor in graph.neighbors(x):
        distribution[node_to_index[neighbor]] = 1. if edge_weight_label is None else \
            graph.edges[x, neighbor][edge_weight_label] if edge_weight_label in graph.edges[x, neighbor] else \
            0.

    distribution_sum = sum(distribution)
    if distribution_sum != 0.:
        distribution *= (1 - alpha) / distribution_sum
        distribution[node_to_index[x]] = alpha
    else:
        distribution[node_to_index[x]] = 1.

    return distribution

def get_augmented_distribution(
    graph: nx.Graph, graph_augmented: nx.Graph,
    node_to_index_augmented: typing.Dict[typing.Any, int],
    x: int, y: int, alpha: float,
    edge_weight_label: typing.Optional[str] = None
) -> npt.NDArray[np.float64]:
    distribution = np.zeros(len(graph_augmented.nodes))

    for neighbor in graph.neighbors(x):
        if neighbor != y:
            distribution[node_to_index_augmented[neighbor]] = 1. if edge_weight_label is None else \
                graph.edges[x, neighbor][edge_weight_label] if edge_weight_label in graph.edges[x, neighbor] else \
                0.
    distribution[node_to_index_augmented[(x, y)]] = 1. if edge_weight_label is None else \
            graph.edges[x, y][edge_weight_label] if edge_weight_label in graph.edges[x, y] else \
            0.

    distribution_sum = sum(distribution)
    if distribution_sum != 0.:
        distribution *= (1 - alpha) / distribution_sum
        distribution[node_to_index_augmented[x]] = alpha
    else:
        distribution[node_to_index_augmented[x]] = 1.

    return distribution

def ricci_curvature_optimal_transport(
    graph: nx.Graph,
    *,
    edge_weight_label: typing.Optional[str] = None,
    edge_distance_label: typing.Optional[str] = None,
    alpha: float = 0.9999,
    use_augmented_graph: bool = True
) -> typing.Dict[typing.Tuple[typing.Any, typing.Any], float]:
    """
    Compute the Ricci curvature for each edge in a graph.

    This function uses optimal transport to compute the graph Ricci
    curvature for each edge in a graph.

    Optionally, this function takes in edge weights and edge distances
    (stored as graph attributes) which are used to calculate distances
    and distributions for the earth mover's distance computaiton.

    The alpha parameter controls how much mass is placed on the central
    node of the distributions. Typically, a value closer to 1 is better.
    TODO: This might not be necessarily true for the augmented graph
    strategy, but further investigation is needed (low priority).

    If desired, an augmented graph is used in the computation, meaning
    this method differs from the original graph Ricci curvature.
    """

    # Create mappings between numerical IDs and node names
    index_to_node = [node for node in graph.nodes]
    node_to_index = {node: index for index, node in enumerate(index_to_node)}
    n_nodes = len(index_to_node)

    # Relabel the graph so it has integer node names
    graph = nx.relabel_nodes(graph, node_to_index)
    distance_matrix = graph_distance.compute_distance_matrix(graph, edge_distance_label)

    if use_augmented_graph:
        graph_augmented = get_augmented_graph(graph, node_to_index)
        index_to_node_augmented = [node for node in graph_augmented.nodes]
        node_to_index_augmented = {node: index for index, node in enumerate(index_to_node_augmented)}
        distance_matrix_augmented = compute_augmented_distances(graph_augmented, distance_matrix)
    else:
        graph_augmented = graph
        node_to_index_augmented = node_to_index
        distance_matrix_augmented = distance_matrix

    ricci_curvatures = {}
    for source, destination in graph.edges:
        if use_augmented_graph:
            distribution_source = get_augmented_distribution(
                graph, graph_augmented, node_to_index_augmented,
                source, destination,
                alpha, edge_weight_label
            )
            distribution_destination = get_augmented_distribution(
                graph, graph_augmented, node_to_index_augmented,
                destination, source,
                alpha, edge_weight_label
            )
        else:
            distribution_source = get_distribution(
                graph, node_to_index,
                source, destination,
                edge_weight_label
            )
            distribution_destination = get_distribution(
                graph, node_to_index,
                destination, source,
                edge_weight_label
            )

        transport_distance = ot.emd2(distribution_source, distribution_destination, distance_matrix_augmented if use_augmented_graph else distance_matrix)
        edge_distance = 1. if edge_distance_label is None else graph.edges[source, destination][edge_distance_label]
        ricci_curvatures[index_to_node[source], index_to_node[destination]] = (1. - transport_distance / edge_distance) / (1. - alpha)

    return ricci_curvatures

def ricci_curvature_optimal_transport_old(
    graph: nx.Graph,
    *,
    edge_weight_label: typing.Optional[str] = None,
    edge_distance_label: typing.Optional[str] = None,
    alpha: float = 0.9999
) -> typing.Dict[typing.Tuple[typing.Any, typing.Any], float]:
    """
    Compute the Ricci curvature for each edge in a graph.

    This function uses optimal transport to compute the graph Ricci
    curvature for each edge in a graph.

    This function is kept for legacy purposes. It should not be called
    anywhere.
    """

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

    # Compute the distance matrix, where distances are either
    # * shortest path lengths
    # * expected (round trip) commute times of random walks on the graph
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
