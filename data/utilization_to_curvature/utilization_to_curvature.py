import argparse
import collections
import csv
import itertools
import os
import pathlib
import sys

import networkx as nx
import numpy as np
import ot
from scipy import optimize, sparse

sys.path.insert(0, str(pathlib.PurePath('..', '..', 'src')))
from linear_geodesic_optimization.data import curvature, input_network, utility, tomography


def compute_curvature_optimal_transport(graph: nx.Graph):
    routes = tomography.get_shortest_routes(graph, 'rtt')
    traffic_matrix = tomography.compute_traffic_matrix(graph, routes, 'throughput')

    index_to_node = list(graph.nodes)
    node_to_index = {node: index for index, node in enumerate(graph.nodes)}

    n = graph.number_of_nodes()
    distance_matrix = np.full((n, n), np.inf)
    for source, routes_source in routes.items():
        index_source = node_to_index[source]
        for destination, route in routes_source.items():
            index_destination = node_to_index[destination]

            distance_route = sum([
                graph.edges[(x, y)]['rtt']
                for x, y in itertools.pairwise(route)
            ])

            if distance_route < distance_matrix[index_source, index_destination]:
                distance_matrix[index_source, index_destination] = distance_route
                # distance_matrix[index_destination, index_source] = distance_route  # For symmetrization

    for u, v in graph.edges:
        distribution_u = np.zeros(n)
        distribution_v = np.zeros(n)

        for source, routes_source in routes.items():
            for destination, route in routes_source.items():
                s_index = 0
                while True:
                    if s_index == len(route) or (route[s_index], u) in graph.edges:
                        break
                    s_index += 1
                if s_index == len(route) or route[s_index] == v:
                    continue

                t_index = len(route) - 1
                while True:
                    if t_index == -1 or (v, route[t_index]) in graph.edges:
                        break
                    t_index -= 1
                if t_index == -1 or route[t_index] == u:
                    continue

                if s_index > t_index:
                    continue

                if (source, destination) not in traffic_matrix:
                    continue
                x_p = traffic_matrix[(source, destination)]

                distribution_u[node_to_index[route[s_index]]] += x_p
                distribution_v[node_to_index[route[t_index]]] += x_p

        normalization_factor = np.sum(distribution_u)
        if normalization_factor == 0.:
            print(f'Skipping edge {u} -> {v}')
            continue

        transportation_cost = ot.emd2(
            distribution_u / normalization_factor,
            distribution_v / normalization_factor,
            distance_matrix
        )
        graph.edges[(u, v)]['curvature'] = 1. - transportation_cost / graph.edges[(u, v)]['rtt']

    return graph

def write_undirected_graphml(graph, file_path_output):
    graph_graphml = nx.Graph()
    for node, data in sorted(graph.nodes(data=True)):
        graph_graphml.add_node(
            node,
            lat=data['lat'],
            long=data['long']
        )
    for u, v, data in sorted(graph.edges(data=True)):
        if (u, v) in graph_graphml.edges:
            graph_graphml.edges[u, v]['gcl'] = min(graph_graphml.edges[u, v]['gcl'], data['gcl'])

            if 'curvature' in data:
                graph_graphml.edges[u, v]['ricciCurvature'] = (graph_graphml.edges[u, v]['ricciCurvature'] + data['curvature']) / 2.
        else:
            data_to_add = {
                'gcl': data['gcl']
            }
            if 'curvature' in data:
                data_to_add['ricciCurvature'] = data['curvature']
            graph_graphml.add_edge(u, v, **data_to_add)
    nx.write_graphml(graph_graphml, file_path_output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pobes', '-p', metavar='probes-file', dest='file_path_probes', type=str, required=True)
    parser.add_argument('--links', '-l', metavar='latencies_file', dest='file_path_links', type=str, required=True)
    parser.add_argument('--output', '-o', metavar='output_file', dest='file_path_output', type=str, required=True)
    parser.add_argument('--optimal-transport', '-t', dest='use_optimal_transport', action='store_true')
    parser.add_argument('--symmetrize', '-s', dest='symmetrize', action='store_true')
    args = parser.parse_args()

    file_path_probes = pathlib.PurePath(args.file_path_probes)
    file_path_links = pathlib.PurePath(args.file_path_links)
    file_path_output = pathlib.PurePath(args.file_path_output)
    use_optimal_transport = args.use_optimal_transport
    symmetrize = args.symmetrize

    if not os.path.exists(file_path_probes):
        sys.stderr.write('Probes file does not exist')
        sys.exit(0)
    if not os.path.exists(file_path_links):
        sys.stderr.write('Links file does not exist')
        sys.exit(0)

    graph = input_network.get_graph_from_csvs(
        file_path_probes, file_path_links,
        clustering_distance=np.finfo(np.float64).eps,
        should_compute_curvatures=False,
        directed=True,
        symmetrize=symmetrize
    )
    if use_optimal_transport:
        graph = compute_curvature_optimal_transport(graph)
    else:
        graph.remove_edges_from([
            (u, v)
            for u, v, data in graph.edges(data=True)
            if 'throughput' not in data or data['throughput'] == 0.
        ])
        curvatures = curvature.compute_ricci_curvature(
            graph,
            use_tomography=True,
            edge_distance_label='rtt',
            edge_weight_label='throughput',
        )
        for (source, destination), kappa in curvatures.items():
            graph.edges[(source, destination)]['curvature'] = kappa

    write_undirected_graphml(graph, file_path_output)
