import typing

import networkx as nx
import numpy as np
import numpy.typing as npt
from scipy import sparse
import scipy.sparse.linalg as spla

import linear_geodesic_optimization.graph.utility as graph_utility


def compute_distance_matrix(
    graph: nx.Graph,
    edge_distance_label: typing.Optional[typing.Any] = None
) -> npt.NDArray[np.float64]:
    """
    Compute the graph distance between point pairs in a graph.

    Distances are determined by the edge distance label, with a default
    of 1 if no label is passed in.
    """
    index_to_node = [node for node in graph.nodes]
    node_to_index = {node: index for index, node in enumerate(index_to_node)}
    n_nodes = len(graph.nodes)

    if edge_distance_label is None:
        distances_iterator = nx.all_pairs_shortest_path_length(graph)
    else:
        distances_iterator = nx.all_pairs_dijkstra_path_length(
            graph, weight = edge_distance_label
        )

    distance_matrix = np.zeros((n_nodes, n_nodes))
    for source, distance_dict in distances_iterator:
        index_source = node_to_index[source]
        for destination, distance in distance_dict.items():
            index_destination = node_to_index[destination]
            distance_matrix[index_source, index_destination] = distance

    return distance_matrix

def compute_random_walk_distance_matrix(
    graph: nx.Graph,
    edge_weight_label: typing.Optional[str],
    edge_distance_label: typing.Optional[str] = None,  # TODO: incorporate
) -> npt.NDArray[np.float64]:
    """
    Compute the expected commute time between point pairs in a graph.

    Given a graph, for each pair (u, v) of vertices, compute the
    expected random walk path length going from u to v back to u.

    Random walk transition probabilities are decided by the edge weights
    with the given weight label (by default, a uniform choice between
    edges), and distances are decided by the given distance label (by
    default, all 1).
    """
    index_to_node = [node for node in graph.nodes]
    node_to_index = {node: index for index, node in enumerate(index_to_node)}
    n_nodes = len(graph.nodes)
    n_edges = len(graph.edges)

    distances = np.zeros((n_nodes, n_nodes))

    # TODO: Allow disconnectivity
    # The jth entry in the ith row is the probability of
    # transitioning from the ith state to the jth state
    edge_weights = graph_utility.csc_matrix_from_attribute(graph, edge_weight_label, 0.)
    edge_weights = (edge_weights / edge_weights.sum(1).reshape((-1, 1))).tocsc()

    edge_lengths = graph_utility.csc_matrix_from_attribute(graph, edge_distance_label, 1.)

    weight_dot_length = (edge_weights * edge_lengths).sum(1)

    for destination_index, destination in enumerate(index_to_node):
        # The distance from the destination to the destination is 0. For
        # every other source, the distance is computed by one step
        # analysis. This involves solving a system with n - 1 equations.

        indices = [index for index in range(n_nodes) if index != destination_index]
        a = sparse.eye(n_nodes - 1) - edge_weights[np.ix_(indices, indices)]
        x = spla.spsolve(a, weight_dot_length[destination_index])
        distances[np.ix_((destination_index,), indices)] = x

    return distances + distances.T
