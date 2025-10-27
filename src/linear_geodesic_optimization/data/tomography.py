import collections
import itertools

import networkx as nx
import numpy as np
from scipy import optimize, sparse


def get_shortest_routes(graph, weight_label='latency'):
    # Compute routes between each source-destination pair. Assume shortest
    # path routing for this example.
    return {
        source: nx.single_source_dijkstra_path(graph, source, weight=weight_label)
        for source in graph.nodes
    }

def compute_traffic_matrix(graph, routes, weight_label='throughput'):
    index_to_link_id = []
    link_id_to_index = {}
    traffic_out_per_node = collections.defaultdict(float)
    traffic_in_per_node = collections.defaultdict(float)
    traffic_total = 0.
    for index, (id_source, id_target, throughput) in enumerate(graph.edges.data(weight_label)):
        index_to_link_id.append((id_source, id_target))
        link_id_to_index[(id_source, id_target)] = index

        traffic_out_per_node[id_source] += throughput
        traffic_in_per_node[id_target] += throughput
        traffic_total += throughput

    # This is a scaling of y by the total traffic in the system
    traffic_counts = np.array([
        graph.edges[source_id, target_id][weight_label]
        for source_id, target_id in index_to_link_id
    ])

    # Determine the ordering of the columns of A, ignoring
    # source-destination pairs with no possible traffic
    sources, destinations = zip(*[
        (source, destination)
        for source in graph.nodes
        for destination in graph.nodes
        if (
            source != destination
            and traffic_out_per_node[source] > 0.
            and traffic_in_per_node[destination] > 0.
        )
    ])

    # Also have a mapping to go from source-destination pairs to indices
    source_destination_to_index = {
        (source, destination): index
        for index, (source, destination) in enumerate(zip(sources, destinations))
    }

    # Create the (sparse) traffic matrix and its transpose
    traffic_matrix_data = []
    traffic_matrix_row_ind = []
    traffic_matrix_col_ind = []
    for index, (source, destination) in enumerate(zip(sources, destinations)):
        route = routes[source][destination]
        for link_id in itertools.pairwise(route):
            traffic_matrix_data.append(1)
            traffic_matrix_row_ind.append(link_id_to_index[link_id])
            traffic_matrix_col_ind.append(index)

    traffic_matrix = sparse.csr_matrix(
        (traffic_matrix_data, (traffic_matrix_row_ind, traffic_matrix_col_ind)),
        shape=(len(index_to_link_id), len(sources))
    )
    traffic_matrix_transpose = sparse.csr_matrix(
        (traffic_matrix_data, (traffic_matrix_col_ind, traffic_matrix_row_ind)),
        shape=(len(sources), len(index_to_link_id))
    )

    def loss(xs, lam=0.01):
        errors = traffic_counts - traffic_total * (traffic_matrix @ xs)
        accuracy = errors @ errors

        penalty = 0.
        if lam != 0.:
            for x, source, destination in zip(xs, sources, destinations):
                n_s = traffic_out_per_node[source]
                n_d = traffic_in_per_node[destination]
                if n_s > 0. and n_d > 0. and x != 0.:
                    penalty += x * np.log2(x * traffic_total**2 / (n_s * n_d))

        if np.isinf(lam):
            return penalty

        return accuracy / traffic_total**2 + lam**2 * penalty

    def dif_loss(xs, lam=0.01):
        errors = traffic_counts - traffic_total * (traffic_matrix @ xs)
        dif_accuracy = -2 * traffic_total * traffic_matrix_transpose @ errors

        if lam != 0.:
            dif_penalty = np.array([
                np.log2(x * traffic_total**2 / (n_s * n_d)) + 1 / np.log(2)
                if n_s > 0. and n_d > 0. else 0.
                for x, source, destination in zip(xs, sources, destinations)
                for n_s in (traffic_out_per_node[source],)
                for n_d in (traffic_in_per_node[destination],)
            ])
        else:
            dif_penalty = np.zeros(xs.shape)

        return dif_accuracy / traffic_total**2 + lam**2 * dif_penalty

    # Gravity model
    x_0 = np.array([
        n_s * n_d / traffic_total**2
        for source, destination in zip(sources, destinations)
        for n_s in (traffic_out_per_node[source],)
        for n_d in (traffic_in_per_node[destination],)
    ])
    x_0 = x_0 / np.sum(x_0)

    x_opt, _, _ = optimize.fmin_l_bfgs_b(
        loss, x_0, fprime=dif_loss, args=[0.01],
        # factr=1e1, pgtol=1e-12,  # Tuning parameters for the optimizer
        bounds=[(1e-12, 1.) for _ in x_0]
    )
    x_opt = x_opt / np.sum(x_opt)

    return {
        (source, destination): x
        for source, destination, x in zip(sources, destinations, x_opt)
    }
