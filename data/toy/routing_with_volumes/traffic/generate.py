import copy
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


# A frame is the traffic at a single point in time: a list of
# (route, volume) pairs. The same origin-destination pair may appear
# more than once (e.g., when traffic is split between two routes).
Frame = list[tuple[list[str], float]]


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

def build_graph() -> nx.Graph:
    graph = nx.Graph()

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

    return graph

def write_graph(graph: nx.Graph, frame: Frame, path: pathlib.PurePath):
    data = {
        'nodes': [
            {
                'id': node,
                'latitude': graph.nodes[node]['latitude'],
                'longitude': graph.nodes[node]['longitude'],
            }
            for node in graph.nodes
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
                'volume': volume,
            }
            for route, volume in frame
        ]
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def write_sequence(
    graph: nx.Graph,
    directory_output: pathlib.PurePath,
    alphas: typing.Iterable[float],
    make_frame: typing.Callable[[float], Frame],
):
    """Write `graph_{i}.json` for the frame at each alpha."""
    os.makedirs(directory_output, exist_ok=True)
    for i, alpha in enumerate(alphas):
        write_graph(graph, make_frame(alpha), directory_output / f'graph_{i}.json')

# Frame building helpers

def frame_from_matrix(routes, traffic_matrix) -> Frame:
    return [
        (route, traffic_matrix[route[0]][route[-1]])
        for route in routes
    ]

def scale_pairs(traffic_matrix, pairs, alpha):
    """Scale traffic in both directions between each pair of nodes."""
    traffic_matrix = copy.deepcopy(traffic_matrix)
    for u, v in pairs:
        traffic_matrix[u][v] *= alpha
        traffic_matrix[v][u] *= alpha
    return traffic_matrix

def cluster_pairs(cluster_left, cluster_right):
    return list(itertools.product(cluster_left, cluster_right))

def pairs_through_link(routes, link):
    """Get the endpoints of each route passing through a link."""
    return [
        (route[0], route[-1])
        for route in routes
        if any(
            (u, v) == link or (v, u) == link
            for u, v in itertools.pairwise(route)
        )
    ]

def routes_without_link(graph: nx.Graph, link):
    """Get shortest routes when a link is removed from the graph."""
    graph = graph.copy()
    graph.remove_edge(*link)
    return tomography.get_shortest_routes(graph, 'latency')

def interpolate_routes(routes_before, routes_after, traffic_matrix, alpha) -> Frame:
    """
    Shift traffic from one set of routes to another.

    For each origin-destination pair whose route changes, a fraction
    `1 - alpha` of its traffic uses the old route and a fraction
    `alpha` uses the new one.
    """
    route_after_by_pair = {(route[0], route[-1]): route for route in routes_after}
    frame = []
    for route_before in routes_before:
        source, destination = route_before[0], route_before[-1]
        volume = traffic_matrix[source][destination]
        route_after = route_after_by_pair[source, destination]
        if route_before == route_after:
            frame.append((route_before, volume))
            continue
        for route, fraction in ((route_before, 1. - alpha), (route_after, alpha)):
            if fraction != 0.:
                frame.append((route, fraction * volume))
    return frame

# Examples. Each takes the graph, the shortest routes, and the base
# traffic matrix, and writes a directory of frames.

CLUSTER_LEFT = ['A', 'B', 'C']
CLUSTER_RIGHT = ['J', 'K']
LINK_CRITICAL = ('F', 'G')

def two_clusters(graph, routes, traffic_matrix):
    pairs = cluster_pairs(CLUSTER_LEFT, CLUSTER_RIGHT)
    write_sequence(
        graph, pathlib.PurePath('two_clusters'), np.linspace(1., 3., 9),
        lambda alpha: frame_from_matrix(routes, scale_pairs(traffic_matrix, pairs, alpha)),
    )

def two_clusters_extreme(graph, routes, traffic_matrix):
    pairs = cluster_pairs(CLUSTER_LEFT, CLUSTER_RIGHT)
    write_sequence(
        graph, pathlib.PurePath('two_clusters_extreme'), np.linspace(1., 6., 9),
        lambda alpha: frame_from_matrix(routes, scale_pairs(traffic_matrix, pairs, alpha)),
    )

def die_out(graph, routes, traffic_matrix):
    pairs = pairs_through_link(routes, LINK_CRITICAL)
    write_sequence(
        graph, pathlib.PurePath('die_out'), np.linspace(1., 0., 9),
        lambda alpha: frame_from_matrix(routes, scale_pairs(traffic_matrix, pairs, alpha)),
    )

def two_clusters_die_out(graph, routes, traffic_matrix):
    pairs = cluster_pairs(CLUSTER_LEFT, CLUSTER_RIGHT)
    write_sequence(
        graph, pathlib.PurePath('two_clusters_die_out'), np.linspace(1., 0., 9),
        lambda alpha: frame_from_matrix(routes, scale_pairs(traffic_matrix, pairs, alpha)),
    )

def reroute_link_outage(graph, routes, traffic_matrix):
    # Same routes as ../reroute/link_outage
    routes_after = routes_without_link(graph, LINK_CRITICAL)
    write_sequence(
        graph, pathlib.PurePath('reroute_link_outage'), np.linspace(0., 1., 9),
        lambda alpha: interpolate_routes(routes, routes_after, traffic_matrix, alpha),
    )

EXAMPLES = {
    'two_clusters': two_clusters,
    'two_clusters_extreme': two_clusters_extreme,
    'die_out': die_out,
    'two_clusters_die_out': two_clusters_die_out,
    'reroute_link_outage': reroute_link_outage,
}

def main(names: list[str]):
    """Generate the named examples (or all of them if none are named)."""
    unknown = [name for name in names if name not in EXAMPLES]
    if unknown:
        raise ValueError(f'Unknown examples {unknown}; choose from {list(EXAMPLES)}')

    graph = build_graph()
    routes = tomography.get_shortest_routes(graph, 'latency')
    traffic_matrix = generate_traffic_matrix(graph)
    for name in names or EXAMPLES:
        EXAMPLES[name](graph, routes, traffic_matrix)

if __name__ == '__main__':
    main(sys.argv[1:])
