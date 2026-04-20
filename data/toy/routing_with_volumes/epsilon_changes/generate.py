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
    return [(1. - alpha) * v_i + alpha * v_f for v_i, v_f in zip(initial, final)]

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
                'volume': volume,
            }
            for route, volume in zip(routes, traffic_matrix)
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
    traffic_initial = [
        1. if route[0][0] == route[-1][0] else
        1. if route[0][0] in 'AB' and route[-1][0] in 'AB' else
        2.
        for route in routes
    ]
    traffic_final = [
        1. if route[0][0] == route[-1][0] else
        6. if route[0][0] in 'AB' and route[-1][0] in 'AB' else
        2.
        for route in routes
    ]

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
    traffic_initial = [
        1. if route[0][0] == route[-1][0] else
        1. if route[0][0] in 'AB' and route[-1][0] in 'AB' else
        2.
        for route in routes
    ]
    traffic_final = [
        1. if route[0][0] == route[-1][0] else
        6. if route[0][0] in 'AB' and route[-1][0] in 'AB' else
        2.
        for route in routes
    ]

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

    traffic_initial = [
        1. if route[0][0] == route[-1][0] else
        1. if route[0] in ['A_0', 'A_1', 'A_2', 'A_3'] or route[-1] in ['A_0', 'A_1', 'A_2', 'A_3'] else
        5.
        for route in routes
    ]
    traffic_final = [
        1. if route[0][0] == route[-1][0] else
        5. if route[0] in ['A_0', 'A_1', 'A_2', 'A_3'] or route[-1] in ['A_0', 'A_1', 'A_2', 'A_3'] else
        1.
        for route in routes
    ]

    compute_latencies(graph)
    plot_scenario(graph, routes, traffic_initial, traffic_final, 'parallel_links')
    for i in range(7):
        write_graph(graph, routes, interpolate_traffic(traffic_initial, traffic_final, i / 6.), pathlib.PurePath('parallel_links', f'{i}.json'))

def generate_sinusoid():
    graph = nx.DiGraph()
    add_cluster(
        graph, 8, np.array([-0.25, 0.]), 1/16, 'A', 2 * np.pi / 16
    )
    add_cluster(
        graph, 8, np.array([0.25, 0.]), 1/16, 'B', 2 * np.pi / 16
    )
    graph.add_node(
        'C_0',
        x=0.,
        y=graph.nodes['A_0']['y'],
        latitude=graph.nodes['A_0']['latitude'],
        longitude=0.
    )
    graph.add_node(
        'C_1',
        x=0.,
        y=graph.nodes['A_7']['y'],
        latitude=graph.nodes['A_7']['latitude'],
        longitude=0.
    )
    graph.add_edge('A_0', 'C_0')
    graph.add_edge('C_0', 'A_0')
    graph.add_edge('C_0', 'B_3')
    graph.add_edge('B_3', 'C_0')
    graph.add_edge('A_7', 'C_1')
    graph.add_edge('C_1', 'A_7')
    graph.add_edge('B_4', 'C_1')
    graph.add_edge('C_1', 'B_4')
    graph.add_edge('C_0', 'C_1')
    graph.add_edge('C_1', 'C_0')

    # Intracluster routes
    routes_intercluster = [
        [f'{cluster_label}_{i}', f'{cluster_label}_{j}']
        for cluster_label in ['A', 'B']
        for i in range(8)
        for j in range(8)
        if i != j
    ]
    # Intercluster taking top left and bottom right paths
    routes_up_down = [
        (
            ([f'A_{i}'] if i != 0 else [])
            + ['A_0', 'C_0', 'C_1', 'B_4']
            + ([f'B_{j}'] if j != 4 else [])
        )
        for i in range(8)
        for j in range(8)
    ] + [
        (
            ([f'B_{j}'] if j != 4 else [])
            + ['B_4', 'C_1', 'C_0', 'A_0']
            + ([f'A_{i}'] if i != 0 else [])
        )
        for i in range(8)
        for j in range(8)
    ]
    # Intercluster taking bottom left and top right paths
    routes_down_up = [
        (
            ([f'A_{i}'] if i != 7 else [])
            + ['A_7', 'C_1', 'C_0', 'B_3']
            + ([f'B_{j}'] if j != 3 else [])
        )
        for i in range(8)
        for j in range(8)
    ] + [
        (
            ([f'B_{j}'] if j != 3 else [])
            + ['B_3', 'C_0', 'C_1', 'A_7']
            + ([f'A_{i}'] if i != 7 else [])
        )
        for i in range(8)
        for j in range(8)
    ]
    routes = routes_intercluster + routes_up_down + routes_down_up

    traffic_initial = (
        [1.] * len(routes_intercluster)
        + [5.] * len(routes_up_down)
        + [1.] * len(routes_down_up)
    )
    traffic_final = (
        [1.] * len(routes_intercluster)
        + [1.] * len(routes_up_down)
        + [5.] * len(routes_down_up)
    )

    compute_latencies(graph)
    plot_scenario(graph, routes, traffic_initial, traffic_final, 'sinusoid')
    for i in range(7):
        write_graph(graph, routes, interpolate_traffic(traffic_initial, traffic_final, i / 6.), pathlib.PurePath('sinusoid', f'{i}.json'))


if __name__ == '__main__':
    # generate_cluster_above()
    # generate_clusters_both_sides()
    # generate_parallel_links()
    generate_sinusoid()
    plt.show()
