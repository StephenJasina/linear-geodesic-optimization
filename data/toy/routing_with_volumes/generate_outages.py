import heapq
import itertools
import json
import os
import pathlib
import sys
import typing

import networkx as nx
import numpy as np

sys.path.append(str(pathlib.PurePath('..', '..', '..', 'src')))
from linear_geodesic_optimization.data import utility


class PriorityQueue:
    """
    A simple wrapper for Python's heapq utilities.

    Based on https://docs.python.org/3/library/heapq.html#priority-queue-implementation-notes
    """
    def __init__(self):
        self.pq = []
        self.entry_finder = {}
        self.counter = itertools.count()

    def add(self, value: str, priority):
        'Add a new task or update the priority of an existing task'
        if value in self.entry_finder:
            self.remove(value)
        id = next(self.counter)
        entry = [priority, value, id, True]
        self.entry_finder[value] = entry
        heapq.heappush(self.pq, entry)

    def remove(self, value: str):
        entry = self.entry_finder.pop(value)
        entry[3] = False

    def pop(self) -> typing.Tuple[str, float]:
        'Remove and return the lowest priority task. Raise KeyError if empty.'
        while self.pq:
            priority, value, _, is_valid = heapq.heappop(self.pq)
            if is_valid:
                del self.entry_finder[value]
                return value, priority
        raise KeyError('pop from an empty priority queue')

def compute_routes(graph: nx.Graph, weight_label: typing.Optional[str]=None):
    """
    Run Dijkstra's algorithm.

    This is necessary ensure tie-breaking is always done in the same
    way.
    """
    routes = {}
    for source in graph.nodes:
        tree = {
            node: {
                'distance': np.inf,
                'predecessor': None
            }
            for node in graph.nodes
        }
        tree[source]['distance'] = 0.
        order_visited = []

        queue = PriorityQueue()
        for node in graph.nodes:
            queue.add(node, 0. if node == source else np.inf)
        while len(order_visited) < graph.number_of_nodes():
            node, distance = queue.pop()
            if np.isposinf(distance):
                break
            order_visited.append(node)
            for successor in graph.neighbors(node):
                d_node_successor = graph.edges[node, successor][weight_label] if weight_label is not None else 1.
                distance_candidate = tree[node]['distance'] + d_node_successor
                if distance_candidate < tree[successor]['distance']:
                    tree[successor] = {
                        'distance': distance_candidate,
                        'predecessor': node,
                    }
                    queue.add(successor, distance_candidate)

        routes_from_source = {}
        for node in order_visited:
            predecessor = tree[node]['predecessor']
            if predecessor is None:
                routes_from_source[node] = [node]
            else:
                routes_from_source[node] = routes_from_source[predecessor] + [node]
        routes[source] = routes_from_source
    return routes

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
        'routes': [
            {
                'route': route,
                'volume': traffic_matrix[source][destination]
            }
            for source, routes_source in routes.items()
            for destination, route in routes_source.items()
        ]
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == '__main__':
    directory_output = pathlib.PurePath('graphs_outage')
    os.makedirs(directory_output, exist_ok=True)

    graph = nx.Graph()

    scale = 0.05
    graph.add_node(
        'A',
        longitude=utility.inverse_mercator(x=-scale * (1 + 3**.5) / 2.),
        latitude=utility.inverse_mercator(y=0.)
    )
    graph.add_node(
        'B',
        longitude=utility.inverse_mercator(x=-scale / 2.),
        latitude=utility.inverse_mercator(y=scale / 2.)
    )
    graph.add_node(
        'C',
        longitude=utility.inverse_mercator(x=-scale / 2.),
        latitude=utility.inverse_mercator(y=-scale / 2.)
    )
    graph.add_node(
        'D',
        longitude=utility.inverse_mercator(x=scale * (1 + 3**.5) / 2.),
        latitude=utility.inverse_mercator(y=0.)
    )
    graph.add_node(
        'E',
        longitude=utility.inverse_mercator(x=scale / 2.),
        latitude=utility.inverse_mercator(y=scale / 2.)
    )
    graph.add_node(
        'F',
        longitude=utility.inverse_mercator(x=scale / 2.),
        latitude=utility.inverse_mercator(y=-scale / 2.)
    )

    traffic_matrix = generate_traffic_matrix(graph)

    graph.add_edge('A', 'B', latency=10000000.)
    graph.add_edge('A', 'C', latency=10000000.)
    graph.add_edge('B', 'C', latency=10000000.)
    graph.add_edge('D', 'E', latency=10000000.)
    graph.add_edge('D', 'F', latency=10000000.)
    graph.add_edge('E', 'F', latency=10000000.)
    graph.add_edge('B', 'E', latency=10000000.)
    graph.add_edge('C', 'F', latency=10000000.)
    routes = compute_routes(graph, 'latency')
    write_graph(graph, routes, traffic_matrix, directory_output / 'graph.json')

    for removals in [
        [('A', 'B')],
        [('B', 'C')],
        [('B', 'E')],
        [('A', 'B'), ('B', 'C')],
        [('A', 'B'), ('B', 'E')],
        [('A', 'B'), ('C', 'F')],
        [('A', 'B'), ('D', 'E')],
        [('A', 'B'), ('D', 'F')],
        [('A', 'B'), ('E', 'F')],
        [('B', 'C'), ('B', 'E')],
        [('B', 'C'), ('E', 'F')],
    ]:
        graph_original = graph.copy()
        graph.remove_edges_from(removals)
        routes = compute_routes(graph, 'latency')
        write_graph(graph_original, routes, traffic_matrix, directory_output / f"graph_{'_'.join(''.join(edge) for edge in removals)}.json")
        write_graph(graph, routes, traffic_matrix, directory_output / f"graph_{'_'.join(''.join(edge) for edge in removals)}_alt.json")

        graph = graph_original
