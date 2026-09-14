import copy
import heapq
import itertools
import json
import os
import pathlib
import sys
import typing

import networkx as nx
import numpy as np

sys.path.append(str(pathlib.PurePath('..', '..', '..', '..', 'src')))
from linear_geodesic_optimization.data import utility
from linear_geodesic_optimization.data import tomography


def generate_traffic_matrix(graph: nx.Graph):
    """Generate a traffic matrix using the stable-fP IC model."""
    # Typically between 0.2 and 0.3
    f = 0.25
    # For now, set all P values to 1. More generally log-normally
    # distributed
    p = {node: 1. for node in graph.nodes}
    p_sum = sum(p.values())
    # For now, set all A values to 1
    # TODO: Should this be a parameter
    a = {node: 1. for node in graph.nodes}

    traffic = {}
    for source in graph.nodes:
        traffic_from_source = {}
        for destination in graph.nodes:
            traffic_from_source_to_destination \
                = (f * a[source] * p[destination] + (1 - f) * a[destination] * p[source]) / p_sum
            traffic_from_source[destination] = traffic_from_source_to_destination
        traffic[source] = traffic_from_source
    return traffic

def write_graph(graph: nx.Graph, routes, traffic_matrix, path: pathlib.PurePath):
    index_to_node = list(graph.nodes)
    node_to_index = {node: index for index, node in enumerate(index_to_node)}

    data = {
        'nodes': [
            {
                'id': node,
                'latitude': graph.nodes[node]['latitude'],
                'longitude': graph.nodes[node]['longitude'],
            }
            for node in index_to_node
        ],
        'links': [
            {
                'source_id': source,
                'target_id': destination,
                'rtt': data['latency'],
            }
            for source, destination, data in graph.edges(data=True)
        ] + [
            {
                'source_id': destination,
                'target_id': source,
                'rtt': data['latency'],
            }
            for source, destination, data in graph.edges(data=True)
        ],
        'traffic': [
            {
                'route': route,
                'volume': traffic_matrix[source][destination]
            }
            for route in routes
            for source in (route[0],)
            for destination in (route[-1])
        ]
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == '__main__':
    graph = nx.Graph()

    scale = 0.05
    for node, x, y in [
        ('A', -2., 2.),
        ('B', -3., 1.),
        ('C', -2., 0.),
        ('D', -1., -1.),
        ('E', 0., 0.),
        ('F', -1., 1.),
        ('G', 2., 1.),
        ('H', 1., 0.),
        ('I', 2., -1.),
        ('J', 3., 0.),
        ('K', 3, 1.5),
    ]:
        graph.add_node(
            node,
            longitude=utility.inverse_mercator(x=x/6.),
            latitude=utility.inverse_mercator(y=y/6.),
        )

    traffic_matrix = generate_traffic_matrix(graph)

    for node_a, node_b in [
        ('A', 'B'),
        ('A', 'C'),
        ('A', 'F'),
        ('B', 'C'),
        ('B', 'F'),
        ('C', 'D'),
        ('C', 'F'),
        ('D', 'E'),
        ('E', 'F'),
        ('E', 'H'),
        ('F', 'G'),
        ('G', 'H'),
        ('G', 'J'),
        ('G', 'K'),
        ('H', 'I'),
        ('I', 'J'),
        ('J', 'K'),
    ]:
        graph.add_edge(
            node_a,
            node_b,
            latency=utility.get_GCL(
                (graph.nodes[node_a]['latitude'], graph.nodes[node_a]['longitude']),
                (graph.nodes[node_b]['latitude'], graph.nodes[node_b]['longitude']),
            ),
        )
    routes = tomography.get_shortest_routes(graph, 'latency')

    directory_output = pathlib.PurePath('two_clusters')
    cluster_left = ['A', 'B', 'C']
    cluster_right = ['J', 'K']
    os.makedirs(directory_output, exist_ok=True)
    traffic_matrix_original = traffic_matrix
    for i, alpha in enumerate(np.linspace(1., 3., 9)):
        traffic_matrix = copy.deepcopy(traffic_matrix_original)
        for u in cluster_left:
            for v in cluster_right:
                traffic_matrix[u][v] *= alpha
                traffic_matrix[v][u] *= alpha
        write_graph(graph, routes, traffic_matrix, directory_output / f'graph_{i}.json')
    traffic_matrix = traffic_matrix_original
