import argparse
import collections
import csv
import itertools
import os
import pathlib
import sys

import networkx as nx
import numpy as np
from scipy import optimize, sparse

sys.path.insert(0, str(pathlib.PurePath('..', '..', 'src')))
from linear_geodesic_optimization.data import utility


def cluster_graph(graph: nx.DiGraph):
    # Cluster by co-location
    clusters = collections.defaultdict(list)
    for node, data_node in graph.nodes(data=True):
        latitude = data_node['latitude']
        longitude = data_node['longitude']
        clusters[(latitude, longitude)].append(node)

    # Make mappings from cluster representatives to constituents and
    # vice versa
    representative_to_constituents = {
        min(cluster): cluster
        for cluster in clusters.values()
    }
    constituent_to_representative = {
        constituent: representative
        for representative, constituents in representative_to_constituents.items()
        for constituent in constituents
    }

    graph_new = nx.DiGraph()
    for representative in representative_to_constituents.keys():
        graph_new.add_node(
            representative,
            latitude=graph.nodes[representative]['latitude'],
            longitude=graph.nodes[representative]['longitude']
        )

    for u, v, data_u_v in graph.edges(data=True):
        representative_u = constituent_to_representative[u]
        representative_v = constituent_to_representative[v]

        if representative_u == representative_v:
            continue

        if (representative_u, representative_v) in graph_new.edges:
            # Aggregate with existing data
            graph_new.edges[representative_u, representative_v]['latency'] \
                = min(
                    graph_new.edges[representative_u, representative_v]['latency'],
                    data_u_v['latency']
                )
            graph_new.edges[representative_u, representative_v]['throughput'] \
                = graph_new.edges[representative_u, representative_v]['throughput'] \
                    + data_u_v['latency']
        else:
            graph_new.add_edge(
                representative_u, representative_v,
                latency=data_u_v['latency'],
                throughput=data_u_v['throughput']
            )

    return graph_new

def get_graph(file_path_probes, file_path_links):
    graph = nx.DiGraph()
    with open(file_path_probes, 'r') as file_probes:
        reader = csv.DictReader(file_probes)
        for index, row in enumerate(reader):
            graph.add_node(
                row['id'],
                latitude=float(row['latitude']),
                longitude=float(row['longitude'])
            )

    with open(file_path_links, 'r') as file_links:
        reader = csv.DictReader(file_links)
        for row in reader:
            id_source = row['source_id']
            id_target = row['target_id']

            node_source = graph.nodes[id_source]
            node_target = graph.nodes[id_target]

            graph.add_edge(
                row['source_id'], row['target_id'],
                throughput=float(row['throughput']),
                latency=utility.get_GCL(
                    (node_source['latitude'], node_source['longitude']),
                    (node_target['latitude'], node_target['longitude']),
                )
            )

    graph = cluster_graph(graph)

    return graph

def get_routes(graph):
    # Compute routes between each source-destination pair. Assume shortest
    # path routing for this example.
    return {
        source: nx.single_source_dijkstra_path(graph, source, weight='latency')
        for source in graph.nodes
    }

def compute_tomography(graph):
    index_to_link_id = []
    link_id_to_index = {}
    traffic_out_per_node = collections.defaultdict(float)
    traffic_in_per_node = collections.defaultdict(float)
    traffic_total = 0.
    for index, (id_source, id_target, throughput) in enumerate(graph.edges.data('throughput')):
        index_to_link_id.append((id_source, id_target))
        link_id_to_index[(id_source, id_target)] = index

        traffic_out_per_node[id_source] += throughput
        traffic_in_per_node[id_target] += throughput
        traffic_total += throughput

    # This is a scaling of y by the total traffic in the system
    traffic_counts = np.array([
        graph.edges[source_id, target_id]['throughput']
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
    x_0 = x_0 / sum(x_0)

    x_opt, _, _ = optimize.fmin_l_bfgs_b(
        loss, x_0, fprime=dif_loss, args=[0.01],
        # factr=1e1, pgtol=1e-12,  # Tuning parameters for the optimizer
        bounds=[(1e-12, 1.) for _ in x_0]
    )

    return {
        (source, destination): x
        for source, destination, x in zip(sources, destinations, x_opt)
    }

def compute_curvature(graph, tomography):
    # Both of these are mappings from s-t pairs to their associated values
    # (denominator and numerator)
    x_sum = collections.defaultdict(float)
    xd_sum = collections.defaultdict(float)

    for source, routes_source in routes.items():
        for destination, route in routes_source.items():
            if (source, destination) not in tomography:
                # In this case, x_p = 0
                continue

            x_p = tomography[(source, destination)]
            for index_s, s in enumerate(route[:-1]):
                d_p_s_t = 0.
                for t_previous, t in itertools.pairwise(route[index_s:]):
                    d_p_s_t += graph.edges[t_previous, t]['latency']

                    x_sum[(s, t)] += x_p
                    xd_sum[(s, t)] += x_p * d_p_s_t

    transportation_costs = {}
    for u, v in graph.edges:
        denominator = 0.
        numerator = 0.
        for s in graph.predecessors(u):
            if s == v:
                continue
            for t in graph.successors(v):
                if t == u:
                    continue

                denominator += x_sum[(s, t)]
                numerator += xd_sum[(s, t)]
        transportation_costs[(u, v)] = numerator / denominator if denominator != 0. else 3. * graph.edges[u, v]['latency']
        if denominator == 0.:
            sys.stderr.write(f'Manually setting curvature of {u} -> {v} to {-2.}\n')

    for u, v in graph.edges:
        graph.edges[u, v]['curvature'] = 1. - transportation_costs[(u, v)] / graph.edges[u, v]['latency']

    return graph

def write_undirected_graphml(graph, file_path_output):
    graph_graphml = nx.Graph()
    for node, data in graph.nodes(data=True):
        graph_graphml.add_node(
            node,
            lat=data['latitude'],
            long=data['longitude']
        )
    for u, v, data in graph.edges(data=True):
        if (u, v) in graph_graphml.edges:
            graph_graphml.edges[u, v]['gcl'] = min(graph_graphml.edges[u, v]['gcl'], data['latency'])
            graph_graphml.edges[u, v]['ricciCurvature'] = (graph_graphml.edges[u, v]['ricciCurvature'] + data['curvature']) / 2.
        else:
            graph_graphml.add_edge(
                u, v,
                gcl=data['latency'],
                ricciCurvature=data['curvature']
            )
    nx.write_graphml(graph_graphml, file_path_output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pobes', '-p', metavar='probes-file', dest='file_path_probes', type=str, required=True)
    parser.add_argument('--links', '-l', metavar='latencies_file', dest='file_path_links', type=str, required=True)
    parser.add_argument('--output', '-o', metavar='output_file', dest='file_path_output', type=str, required=True)
    args = parser.parse_args()

    file_path_probes = pathlib.PurePath(args.file_path_probes)
    file_path_links = pathlib.PurePath(args.file_path_links)
    file_path_output = pathlib.PurePath(args.file_path_output)

    if not os.path.exists(file_path_probes):
        sys.stderr.write('Probes file does not exist')
        sys.exit(0)
    if not os.path.exists(file_path_links):
        sys.stderr.write('Links file does not exist')
        sys.exit(0)

    graph = get_graph(file_path_probes, file_path_links)
    routes = get_routes(graph)
    tomography = compute_tomography(graph)
    graph = compute_curvature(graph, tomography)
    write_undirected_graphml(graph, file_path_output)
