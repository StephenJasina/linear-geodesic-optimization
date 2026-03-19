import itertools
import json
import os
import pathlib
import sys
import typing

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt

sys.path.append(str(pathlib.PurePath('..', '..', '..', '..', 'src')))
from linear_geodesic_optimization.data import tomography, utility
from visualize import plot_scenario

def add_cluster(graph: nx.DiGraph, size: int, center: npt.NDArray[np.float64], radius: float, name: str, theta_offset: float=0.):
    """
    Add a complete subgraph of nodes.

    The nodes of the cluster lie on a circle of radius `radius` centered
    at `center` (stored into the graph with attributes `'x'` and `'y'`).
    The nodes have names of the format `name_<index>`, and the cluster
    can optionally be rotated by `theta_offset` radians
    counterclockwise.
    """
    node_names = []
    for index in range(size):
        node_name = f'{name}_{index}'
        node_names.append(node_name)
        x = center[0] + radius * np.cos(2 * np.pi * index / size + theta_offset)
        y = center[1] + radius * np.sin(2 * np.pi * index / size + theta_offset)
        longlat_pair = utility.inverse_mercator(x, y)
        graph.add_node(
            node_name,
            x=x, y=y, longitude=longlat_pair[0], latitude=longlat_pair[1]
        )
    for node_a in node_names:
        for node_b in node_names:
            if node_a == node_b:
                continue
            graph.add_edge(node_a, node_b)

def compute_latencies(graph: nx.DiGraph):
    for node_a, node_b in graph.edges:
        # latlong_a = (graph.nodes[node_a]['latitude'], graph.nodes[node_a]['longitude'])
        # latlong_b = (graph.nodes[node_b]['latitude'], graph.nodes[node_b]['longitude'])
        # graph.edges[node_a, node_b]['latency'] = utility.get_GCL(latlong_a, latlong_b)
        xy_a = np.array([graph.nodes[node_a]['x'], graph.nodes[node_a]['y']])
        xy_b = np.array([graph.nodes[node_b]['x'], graph.nodes[node_b]['y']])
        graph.edges[node_a, node_b]['latency'] = np.linalg.norm(xy_a - xy_b)

def interpolate_traffic(initial, final, alpha):
    keys = {}
    for source, traffic_source in initial.items():
        keys_source = set()
        for destination in traffic_source:
            keys_source.add(destination)
        keys[source] = keys_source
    for source, traffic_source in final.items():
        if source in keys:
            keys_source = keys[source]
        else:
            keys_source = set()
        for destination in traffic_source:
            keys_source.add(destination)
        keys[source] = keys_source

    traffic = {}
    for source, keys_source in keys.items():
        traffic_source = {}
        for destination in keys_source:
            if source in initial and destination in initial[source]:
                traffic_initial = initial[source][destination]
            else:
                traffic_initial = 0.
            if source in final and destination in final[source]:
                traffic_final = final[source][destination]
            else:
                traffic_final = 0.
            traffic_source[destination] = (1. - alpha) * traffic_initial + alpha * traffic_final
        traffic[source] = traffic_source
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
        ],
        'traffic': [
            {
                'route': route,
                'volume': traffic_matrix[route[0]][route[-1]]
            }
            for route in routes
        ]
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def generate_cluster_above():
    graph = nx.DiGraph()
    add_cluster(
        graph, 8, np.array([-0.25, 0.]), 1/16, 'A'
    )
    add_cluster(
        graph, 8, np.array([0.25, 0.]), 1/16, 'B'
    )
    add_cluster(
        graph, 8, np.array([0., 0.2]), 1/16, 'C', 2 * np.pi / 16
    )
    graph.add_edge('A_0', 'B_4')
    graph.add_edge('B_4', 'A_0')
    graph.add_edge('A_1', 'C_4')
    graph.add_edge('C_4', 'A_1')
    graph.add_edge('B_3', 'C_7')
    graph.add_edge('C_7', 'B_3')
    routes = tomography.get_shortest_routes(graph)
    traffic_initial = {
        source: {
            destination: (
                1. if source[0] == destination[0] else
                1. if source[0] in 'AB' and destination[0] in 'AB' else
                2.
            )
            for destination in graph.nodes
        }
        for source in graph.nodes
    }
    traffic_final = {
        source: {
            destination: (
                1. if source[0] == destination[0] else
                6. if source[0] in 'AB' and destination[0] in 'AB' else
                2.
            )
            for destination in graph.nodes
        }
        for source in graph.nodes
    }

    compute_latencies(graph)
    plot_scenario(graph, routes, traffic_initial, traffic_final, 'cluster_above')
    for i in range(7):
        write_graph(graph, routes, interpolate_traffic(traffic_initial, traffic_final, i / 6.), pathlib.PurePath('cluster_above', f'{i}.json'))

def generate_clusters_both_sides():
    graph = nx.DiGraph()
    add_cluster(
        graph, 8, np.array([-0.25, 0.]), 1/16, 'A'
    )
    add_cluster(
        graph, 8, np.array([0.25, 0.]), 1/16, 'B'
    )
    add_cluster(
        graph, 8, np.array([0., 0.2]), 1/16, 'C', 2 * np.pi / 16
    )
    add_cluster(
        graph, 8, np.array([0., -0.2]), 1/16, 'D', 2 * np.pi / 16
    )
    graph.add_edge('A_0', 'B_4')
    graph.add_edge('B_4', 'A_0')
    graph.add_edge('A_1', 'C_4')
    graph.add_edge('C_4', 'A_1')
    graph.add_edge('B_3', 'C_7')
    graph.add_edge('C_7', 'B_3')
    graph.add_edge('A_7', 'D_3')
    graph.add_edge('D_3', 'A_7')
    graph.add_edge('B_5', 'D_0')
    graph.add_edge('D_0', 'B_5')
    routes = tomography.get_shortest_routes(graph)
    traffic_initial = {
        source: {
            destination: (
                1. if source[0] == destination[0] else
                1. if source[0] in 'AB' and destination[0] in 'AB' else
                # 1. if source in ['A_0', 'B_4'] and destination in ['A_0', 'B_4'] else
                2.
            )
            for destination in graph.nodes
        }
        for source in graph.nodes
    }
    traffic_final = {
        source: {
            destination: (
                1. if source[0] == destination[0] else
                6. if source[0] in 'AB' and destination[0] in 'AB' else
                # 600. if source in ['A_0', 'B_4'] and destination in ['A_0', 'B_4'] else
                2.
            )
            for destination in graph.nodes
        }
        for source in graph.nodes
    }

    compute_latencies(graph)
    plot_scenario(graph, routes, traffic_initial, traffic_final, 'clusters_both_sides')
    for i in range(7):
        write_graph(graph, routes, interpolate_traffic(traffic_initial, traffic_final, i / 6.), pathlib.PurePath('clusters_both_sides', f'{i}.json'))

def generate_parallel_links():
    graph = nx.DiGraph()
    add_cluster(
        graph, 8, np.array([-0.25, 0.]), 1/16, 'A', 2 * np.pi / 16
    )
    add_cluster(
        graph, 8, np.array([0.25, 0.]), 1/16, 'B', 2 * np.pi / 16
    )
    graph.add_edge('A_0', 'B_3')
    graph.add_edge('B_3', 'A_0')
    graph.add_edge('A_7', 'B_4')
    graph.add_edge('B_4', 'A_7')

    routes = []
    # Routes within clusters are direct
    for i in range(8):
        for j in range(8):
            if i == j:
                continue
            routes.append([f'A_{i}', f'A_{j}'])
            routes.append([f'B_{i}', f'B_{j}'])
    # Half the routes between clusters should take the top edge
    for i in range(4):
        for j in range(8):
            route = [f'A_{i}']
            if i != 0:
                route.append('A_0')
            if j != 3:
                route.append('B_3')
            route.append(f'B_{j}')
            routes.append(route)
            routes.append(list(reversed(route)))
    # Half the routes between clusters should take the bottom edge
    for i in range(4, 8):
        for j in range(8):
            route = [f'A_{i}']
            if i != 7:
                route.append('A_7')
            if j != 4:
                route.append('B_4')
            route.append(f'B_{j}')
            routes.append(route)
            routes.append(list(reversed(route)))

    traffic_initial = {
        source: {
            destination: (
                1. if source[0] == destination[0] else
                1. if source in ['A_0', 'A_1', 'A_2', 'A_3'] or destination in ['A_0', 'A_1', 'A_2', 'A_3'] else
                5.
            )
            for destination in graph.nodes
        }
        for source in graph.nodes
    }
    traffic_final = {
        source: {
            destination: (
                1. if source[0] == destination[0] else
                5. if source in ['A_0', 'A_1', 'A_2', 'A_3'] or destination in ['A_0', 'A_1', 'A_2', 'A_3'] else
                1.
            )
            for destination in graph.nodes
        }
        for source in graph.nodes
    }

    compute_latencies(graph)
    plot_scenario(graph, routes, traffic_initial, traffic_final, 'parallel_links')
    for i in range(7):
        write_graph(graph, routes, interpolate_traffic(traffic_initial, traffic_final, i / 6.), pathlib.PurePath('parallel_links', f'{i}.json'))

if __name__ == '__main__':
    generate_cluster_above()
    generate_clusters_both_sides()
    generate_parallel_links()
    plt.show()
