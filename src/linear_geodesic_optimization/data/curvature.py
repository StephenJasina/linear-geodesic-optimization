import collections
import itertools
import typing

import networkx as nx
import numpy as np
import numpy.typing as npt
import ot
from scipy import optimize, sparse

from linear_geodesic_optimization.graph import distance as graph_distance
from linear_geodesic_optimization.data import tomography


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
      in the "direction" of node j

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
    as a subspace

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
    x: int, alpha: float,
    edge_weight_label: typing.Optional[str] = None
) -> npt.NDArray[np.float64]:
    distribution = np.zeros(len(graph.nodes))

    for neighbor in graph.neighbors(x):
        distribution[neighbor] = 1. if edge_weight_label is None else \
            graph.edges[x, neighbor][edge_weight_label] if edge_weight_label in graph.edges[x, neighbor] else \
            0.

    distribution_sum = sum(distribution)
    if distribution_sum != 0.:
        distribution *= (1 - alpha) / distribution_sum
        distribution[x] = alpha
    else:
        distribution[x] = 1.

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

def compute_ricci_curvature_from_traffic(
    graph: nx.Graph, routes, traffic,
    edge_distance_label='latency',
    use_optimal_transport=False
):
    # Throughout, have some special variables for if we want to use
    # optimal transport to compute the data flow. This places masses on
    # relevant predecessors and successors, then discards routing
    # information.
    if use_optimal_transport:
        node_to_index = {node: index for index, node in enumerate(graph.nodes)}
        distance_matrix = graph_distance.compute_distance_matrix(graph, edge_distance_label)
    else:
        node_to_index = None
        distance_matrix = None

    ricci_curvatures = {}
    edges = [(u, v) for u, v in graph.edges]
    if not isinstance(graph, nx.DiGraph):
        edges += [(v, u) for u, v in graph.edges]
    for u, v in edges:
        d_u_v = graph.edges[u, v][edge_distance_label] if edge_distance_label is not None else 1.

        # Some running totals used to compute transportation costs. In
        # most cases, we should only expect to use the s_t variant, but
        # we compute all of them to avoid having to use more for loops.
        denominator_p_s = 0.
        numerator_p_s = 0.
        denominator_p_v = 0.
        numerator_p_v = 0.
        denominator_u_s = 0.
        numerator_u_s = 0.
        denominator_u_v = 0.
        numerator_u_v = 0.

        if use_optimal_transport:
            distribution_u_p_s = np.zeros(graph.number_of_nodes())
            distribution_v_p_s = np.zeros(graph.number_of_nodes())
            distribution_u_p_v = np.zeros(graph.number_of_nodes())
            distribution_v_p_v = np.zeros(graph.number_of_nodes())
            distribution_u_u_s = np.zeros(graph.number_of_nodes())
            distribution_v_u_s = np.zeros(graph.number_of_nodes())
            distribution_u_u_v = np.zeros(graph.number_of_nodes())
            distribution_v_u_v = np.zeros(graph.number_of_nodes())
        else:
            distribution_u_p_s = None
            distribution_v_p_s = None
            distribution_u_p_v = None
            distribution_v_p_v = None
            distribution_u_u_s = None
            distribution_v_u_s = None
            distribution_u_u_v = None
            distribution_v_u_v = None

        # Find which routes are relevant
        for route, traffic_route in zip(routes, traffic):
            source = route[0]
            destination = route[-1]
            if traffic_route == 0.:
                # In this case, nothing would be accumulated, so we can
                # skip investigating the route
                continue

            has_p_something = False
            has_u_something = False

            has_p_s = False
            has_p_v = False
            has_u_s = False
            has_u_v = False

            for node in route:
                if node != v and (node, u) in graph.edges:
                    has_p_something = True
                if node == u:
                    has_u_something = True

                if has_p_something:
                    if node != u and (v, node) in graph.edges:
                        has_p_s = True
                    if node == v:
                        has_p_v = True
                if has_u_something:
                    if node != u and (v, node) in graph.edges:
                        has_u_s = True
                    if node == v:
                        has_u_v = True

            if has_p_s or has_p_v:
                has_u_s = False
                has_u_v = False
            if has_p_s or has_u_s:
                has_p_v = False
                has_u_v = False

            # Flags for whether we use fall under edge cases
            has_p = has_p_s or has_p_v
            has_s = has_p_s or has_u_s

            # At this point, at most one of has_p_s, has_p_v,
            # has_u_s, and has_u_v is True
            if not (has_p_s or has_p_v or has_u_s or has_u_v):
                # Irrelevant route in this case
                continue

            distance_route = np.inf

            p = None
            s = None

            # Check all length-0 segments of the route
            for node in route:
                is_left = (has_p and node != v and (node, u) in graph.edges) or (not has_p and node == u)
                is_right = (has_s and node!= u and (v, node) in graph.edges) or (not has_s and node == v)
                if is_left and is_right:
                    distance_route = 0.
                    p = node
                    s = node
                    break

            # Check all non-length-0 segments of the route
            if distance_route != 0.:
                distance_route_candidate = np.inf
                p_candidate = None
                for a, b in itertools.pairwise(route):
                    d_a_b = graph.edges[a, b][edge_distance_label] if edge_distance_label is not None else 1.

                    if (has_p and a != v and (a, u) in graph.edges) or (not has_p and a == u):
                        distance_route_candidate = 0.
                        p_candidate = a

                    distance_route_candidate += d_a_b

                    if (has_s and b != u and (v, b) in graph.edges) or (not has_s and b == v):
                        if distance_route_candidate < distance_route:
                            distance_route = distance_route_candidate
                            p = p_candidate
                            s = b

            if use_optimal_transport:
                if has_p:
                    if has_s:
                        distribution_u_p_s[node_to_index[p]] += traffic_route
                        distribution_v_p_s[node_to_index[s]] += traffic_route
                    else:
                        distribution_u_p_v[node_to_index[p]] += traffic_route
                        distribution_v_p_v[node_to_index[s]] += traffic_route
                else:
                    if has_s:
                        distribution_u_u_s[node_to_index[p]] += traffic_route
                        distribution_v_u_s[node_to_index[s]] += traffic_route
                    else:
                        distribution_u_u_v[node_to_index[p]] += traffic_route
                        distribution_v_u_v[node_to_index[s]] += traffic_route

            if has_p:
                if has_s:
                    denominator_p_s += traffic_route
                    numerator_p_s += traffic_route * distance_route
                else:
                    denominator_p_v += traffic_route
                    numerator_p_v += traffic_route * (distance_route + d_u_v)
            else:
                if has_s:
                    denominator_u_s += traffic_route
                    numerator_u_s += traffic_route * (distance_route + d_u_v)
                else:
                    denominator_u_v += traffic_route
                    numerator_u_v += traffic_route * (distance_route + 2 * d_u_v)

        if use_optimal_transport:
            if denominator_p_s != 0.:
                transportation_cost = ot.emd2(
                    distribution_u_p_s,
                    distribution_v_p_s,
                    distance_matrix
                ) / denominator_p_s
            elif denominator_p_v != 0.:
                transportation_cost = ot.emd2(
                    distribution_u_p_v,
                    distribution_v_p_v,
                    distance_matrix
                ) / denominator_p_v
            elif denominator_u_s != 0.:
                transportation_cost = ot.emd2(
                    distribution_u_u_s,
                    distribution_v_u_s,
                    distance_matrix
                ) / denominator_u_s
            elif denominator_u_v != 0.:
                transportation_cost = ot.emd2(
                    distribution_u_u_v,
                    distribution_v_u_v,
                    distance_matrix
                ) / denominator_u_v
            else:
                continue
        else:
            # if denominator_p_s != 0.:
            #     # Prioritize the case where we have data describing
            #     # transportation between neighborhoods of u and v
            #     transportation_cost = numerator_p_s / denominator_p_s
            # elif denominator_p_v != 0.:
            #     # If that data doesn't exist, check whether we have routes
            #     # from u's neighborhood to v
            #     transportation_cost = numerator_p_v / denominator_p_v
            # elif denominator_u_s != 0.:
            #     # If that data doesn't exist, check whether we have routes
            #     # from u to v's neighborhood
            #     transportation_cost = numerator_u_s / denominator_u_s
            # elif denominator_u_v != 0.:
            #     # If that data doesn't exist, check whether we have routes
            #     # from u to v
            #     transportation_cost = numerator_u_v / denominator_u_v
            # else:
            #     # If we get here, we don't have enough information to
            #     # compute the curvature. For now, let's just not set the
            #     # curvature to anything
            #     continue
            transportation_cost = (numerator_p_s + numerator_p_v + numerator_u_s + numerator_u_v) / (denominator_p_s + denominator_p_v + denominator_u_s + denominator_u_v)
        ricci_curvatures[(u, v)] = 1. - transportation_cost / d_u_v

    return ricci_curvatures

def compute_ricci_curvature(
    graph: nx.Graph,
    *,
    edge_weight_label: typing.Optional[str] = None,
    edge_distance_label: typing.Optional[str] = None,
    alpha: float = 0.,
    use_augmented_graph: bool = False,
    use_tomography: bool = False
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

    ricci_curvatures = {}
    if use_augmented_graph:
        graph_augmented = get_augmented_graph(graph, node_to_index)
        index_to_node_augmented = [node for node in graph_augmented.nodes]
        node_to_index_augmented = {node: index for index, node in enumerate(index_to_node_augmented)}
        distance_matrix = compute_augmented_distances(graph_augmented, distance_matrix)

        for source, destination in graph.edges:
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

            transportation_cost = ot.emd2(
                distribution_source,
                distribution_destination,
                distance_matrix
            )
            edge_distance = 1. if edge_distance_label is None else graph.edges[source, destination][edge_distance_label]
            ricci_curvatures[index_to_node[source], index_to_node[destination]] = (1. - transportation_cost / edge_distance) / (1. - alpha)
    elif use_tomography:
        routes_dict = tomography.get_shortest_routes(graph, edge_distance_label=edge_distance_label)
        traffic_matrix_dict = tomography.compute_traffic_matrix(graph, routes_dict, edge_weight_label)
        routes = [routes_dict[s][d] for s, d in traffic_matrix_dict]
        traffic_matrix = list(traffic_matrix_dict.values())
        ricci_curvatures = {
            (index_to_node[source], index_to_node[destination]): curvature
            for (source, destination), curvature in compute_ricci_curvature_from_traffic(
                graph, routes, traffic_matrix, edge_distance_label
            ).items()
        }
    else:
        for source, destination in graph.edges:
            distribution_source = get_distribution(
                graph, source,
                alpha, edge_weight_label
            )
            distribution_destination = get_distribution(
                graph, destination,
                alpha, edge_weight_label
            )

            transportation_cost = ot.emd2(
                distribution_source,
                distribution_destination,
                distance_matrix
            )
            edge_distance = 1. if edge_distance_label is None else graph.edges[source, destination][edge_distance_label]
            ricci_curvatures[index_to_node[source], index_to_node[destination]] = (1. - transportation_cost / edge_distance) / (1. - alpha)

    return ricci_curvatures
