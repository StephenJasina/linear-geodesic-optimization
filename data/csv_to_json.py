"""File containing utility to generate a json file."""

import argparse
import csv
import json
import os
import pathlib
import sys

import networkx as nx
import numpy as np

sys.path.append(os.path.join('..', 'src'))
from linear_geodesic_optimization.data import input_network, tomography


def write_graph(graph: nx.Graph, routes, traffic, path: pathlib.PurePath):
    index_to_node = list(graph.nodes)
    node_to_index = {node: index for index, node in enumerate(index_to_node)}

    data = {
        'nodes': [
            {
                'id': node,
                'latitude': graph.nodes[node]['lat'],
                'longitude': graph.nodes[node]['long'],
            }
            for node in index_to_node
        ],
        'links': [
            {
                'source_id': source,
                'target_id': destination,
                'rtt': data['rtt'],
            }
            for source, destination, data in graph.edges(data=True)
        ] + ([] if isinstance(graph, nx.DiGraph) else [
            {
                'source_id': destination,
                'target_id': source,
                'rtt': data['rtt'],
            }
            for source, destination, data in graph.edges(data=True)
        ]),
        'traffic': [
            {
                'route': route,
                'volume': traffic
            }
            for route, traffic in zip(routes, traffic)
        ]
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == '__main__':
    # Parse arugments
    parser = argparse.ArgumentParser()
    parser.add_argument('--probes-file', '-p', type=str, required=True,
                        dest='probes_filename', metavar='<filename>',
                        help='Input file containing probes information')
    parser.add_argument('--latencies-file', '-l', type=str, required=True,
                        dest='latencies_filename', metavar='<filename>',
                        help='Input file containing latency information')
    parser.add_argument('--epsilon', '-e', type=float, required=False,
                        dest='epsilon', metavar='<epsilon>',
                        help='Residual threshold')
    parser.add_argument('--clustering-distance', '-c', type=float,
                        required=False, dest='clustering_distance',
                        metavar='<clustering distance>')
    parser.add_argument('--output', '-o', metavar='<filename>',
                        dest='output_filename', required=True)
    args = parser.parse_args()
    probes_filename = args.probes_filename
    latencies_filename = args.latencies_filename
    epsilon = args.epsilon
    if epsilon is None:
        epsilon = np.inf
    clustering_distance = args.clustering_distance
    output_filename = args.output_filename

    graph = input_network.get_graph_from_csvs(
        probes_filename, latencies_filename,
        clustering_distance=clustering_distance,
        should_compute_curvatures=False, directed=True, symmetrize=True
    )

    routes = tomography.get_shortest_routes(graph, 'rtt')
    traffic = tomography.compute_traffic(graph, routes, 'throughput')

    write_graph(graph, routes, traffic, output_filename)
