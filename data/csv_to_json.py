"""File containing utility to generate a json file."""

import argparse
import csv
import json
import os
import pathlib
import sys
import typing

import networkx as nx
import numpy as np

sys.path.append(os.path.join('..', 'src'))
from linear_geodesic_optimization.data import input_network, tomography, utility


def write_graph(
    graph: nx.Graph, path: pathlib.PurePath,
    delays: typing.Dict[typing.Tuple[str, str], float]={},
    routes: typing.List[typing.List[str]]=[],
    traffic: typing.List[float]=[]
):
    index_to_node = list(graph.nodes)
    node_to_index = {node: index for index, node in enumerate(index_to_node)}

    # Assumed delays are symmetric, which is reasonable for RTT
    # measurements
    delays_symmetrized = {}
    # TODO: Should we guarantee that the edges in the graph have
    # associated latencies? If not, remove the next few lines.
    # Somewhat expensive operations (makes copies of the delays and the
    # graph), but should be okay since this is not run frequently
    delays = delays.copy()
    for source, destination in graph.to_directed().edges:
        if (source, destination) in delays or (destination, source) in delays:
            continue
        node_source = graph.nodes[source]
        node_destination = graph.nodes[destination]
        delays[source, destination] = utility.get_GCL(
            (node_source['lat'], node_source['long']),
            (node_destination['lat'], node_destination['long'])
        )
    # Add in the actual measured delays
    for (source, destination), delay in delays.items():
        if (destination, source) in delays:
            delay = min(delay, delays[destination, source])
        delays_symmetrized[source, destination] = delay
        delays_symmetrized[destination, source] = delay

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
            }
            for source, destination, data in graph.edges(data=True)
        ] + ([] if isinstance(graph, nx.DiGraph) else [
            {
                'source_id': destination,
                'target_id': source,
            }
            for source, destination, data in graph.edges(data=True)
        ]),
        'delays': [
            {
                'source_id': source,
                'target_id': destination,
                'rtt': delay,
            }
            for (source, destination), delay in sorted(delays_symmetrized.items())
        ],
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

    write_graph(graph, output_filename, routes=routes, traffic=traffic)
