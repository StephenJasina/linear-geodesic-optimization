import collections
import csv
import itertools
import json
import typing

import networkx as nx
import numpy as np

from linear_geodesic_optimization.data import \
    clustering, \
    curvature, \
    tomography, \
    triangle_inequality_violations as tivs, \
    utility


def minimize_id_removal(rtt_violation_list):
    """Approximately solve the min vertex cover problem."""
    id_counts = {}

    for id_source, id_target in rtt_violation_list:
        id_counts[id_source] = id_counts.get(id_source, 0) + 1
        id_counts[id_target] = id_counts.get(id_target, 0) + 1

    ids_to_remove = set()

    while rtt_violation_list:
        # Find id with maximum count
        max_id = max(id_counts, key=id_counts.get)
        ids_to_remove.add(max_id)

        # Remove all lines containing max_id
        rtt_violation_list = [line for line in rtt_violation_list if max_id not in line[:2]]

        # Reset id_counts for the next iteration
        id_counts = {}
        for line in rtt_violation_list:
            id_counts[line[0]] = id_counts.get(line[0], 0) + 1
            id_counts[line[1]] = id_counts.get(line[1], 0) + 1

    return ids_to_remove

def get_base_graph(probes, links, directed=False, symmetrize=False):
    # Create the graph
    if directed:
        graph = nx.DiGraph()
    else:
        graph = nx.Graph()

    # Create the RTT violation list
    rtt_violation_list = []

    # Get the vertecies
    lat_min = np.inf
    lat_max = -np.inf
    long_min = np.inf
    long_max = -np.inf
    for probe in probes:
        lat = float(probe['latitude'])
        long = float(probe['longitude'])
        graph.add_node(
            probe['id'],
            lat=lat, long=long
        )
        if 'city' in probe:
            graph.nodes[probe['id']]['city'] = probe['city']
        if 'country' in probe:
            graph.nodes[probe['id']]['country'] = probe['country']

        # TODO: Make this more robust (in the case longitude "wraps around")
        lat_min = min(lat_min, lat)
        lat_max = max(lat_max, lat)
        long_min = min(long_min, long)
        long_max = max(long_max, long)

    # Store graph bounding box
    graph.graph['lat_min'] = lat_min
    graph.graph['lat_max'] = lat_max
    graph.graph['long_min'] = long_min
    graph.graph['long_max'] = long_max

    # Get the edges
    for link in links:
        id_source = link['source_id']
        id_target = link['target_id']
        if symmetrize:
            id_source, id_target = min(id_source, id_target), max(id_source, id_target)

        # Skip self loops
        if id_source == id_target:
            continue

        # Skip measurements we're missing the nodes for
        if id_source not in graph.nodes or id_target not in graph.nodes:
            continue

        lat_source = graph.nodes[id_source]['lat']
        long_source = graph.nodes[id_source]['long']
        lat_target = graph.nodes[id_target]['lat']
        long_target = graph.nodes[id_target]['long']
        gcl = utility.get_GCL(
            [lat_source, long_source],
            [lat_target, long_target]
        )

        if 'rtt' in link:
            rtt = link['rtt']
            # Skip edges with no measured RTT
            if rtt == '':
                continue
            rtt = float(rtt)
        else:
            # If rtt isn't available at all, just add every edge (i.e.,
            # assume the input contains connectivity information)
            rtt = gcl

        throughput = 0.
        if 'throughput' in link and link['throughput'] != '':
            throughput = float(link['throughput'])

        # Check how often the difference is larger than 0
        # TODO: Should this be more lenient?
        if rtt - gcl < 0:
            rtt_violation_list.append((id_source, id_target))

        if (id_source, id_target) in graph.edges:
            data_edge = graph.edges[id_source,id_target]

            # If there is multiple sets of RTT data for a single
            # edge, only pay attention to the minimal one
            if rtt < data_edge['rtt']:
                data_edge['rtt'] = rtt

            # Aggregate all recorded throughputs
            data_edge['throughput'] += throughput
        else:
            edge_data = {
                'rtt': rtt,
                'gcl': gcl,
                'throughput': throughput
            }
            graph.add_edge(id_source, id_target, **edge_data)

    # TODO: Make this less aggressive
    # TODO: Make this a flag
    # Delete nodes with inconsistent geolocation
    # for node in minimize_id_removal(rtt_violation_list):
    #     graph.remove_node(node)

    if directed and symmetrize:
        graph.add_edges_from([
            (id_target, id_source, data)
            for id_source, id_target, data in graph.edges(data=True)
        ])

    return graph

def threshold_graph(graph, epsilon):
    edges_to_delete = []
    for u, v, d in graph.edges(data=True):
        if 'rtt' not in d:
            continue
        rtt = d['rtt']
        gcl = d['gcl']

        if rtt - gcl > epsilon:
            edges_to_delete.append((u, v))

    for u, v in edges_to_delete:
        graph.remove_edge(u, v)

    return graph

def cluster_graph(graph, distance):
    circumference_earth = 40075016.68557849

    graph = clustering.cluster_graph(
        graph,
        distance / circumference_earth
    )

    return graph

def remove_tivs(graph):
    triangles = tivs.compute_triangles(graph)
    goodnesses = tivs.compute_goodnesses(triangles)
    violating_triangles = [
        triangle
        for triangle, goodness in goodnesses.items()
        if goodness > 1.
    ]
    long_edges = tivs.get_long_edges(graph, violating_triangles)

    for u, v in long_edges:
        graph.remove_edge(u, v)

    return graph

def compute_ricci_curvatures(
    graph: nx.Graph,
    alpha: float=0.,
    weight_label: typing.Optional[str]=None,
    routes=None, traffic_matrix=None, force_optimal_transport=False
):
    ricci_curvatures = {}
    if routes is not None and traffic_matrix is not None:
        ricci_curvatures = curvature.compute_ricci_curvature_from_traffic_matrix(
            graph, routes, traffic_matrix, 'rtt', force_optimal_transport
        )
    else:
        ricci_curvatures = curvature.compute_ricci_curvature(
            graph, alpha=alpha, edge_weight_label=weight_label, use_augmented_graph=False
        )
    for (source, destination), ricci_curvature in ricci_curvatures.items():
        if 'ricciCurvature' in graph.edges[source, destination]:
            graph.edges[source, destination]['ricciCurvature'] = (graph.edges[source, destination]['ricciCurvature'] + ricci_curvature) / 2.
        else:
            graph.edges[source, destination]['ricciCurvature'] = ricci_curvature

    return graph

def get_graph(
    probes, links,
    *,
    epsilon=None,
    clustering_distance=None,
    should_remove_tivs=False,
    should_include_latencies=False,
    should_compute_curvatures=True,
    ricci_curvature_alpha=0.,
    ricci_curvature_weight_label=None,
    directed=False, symmetrize=False,
    routes=None, traffic_matrix=None, force_optimal_transport=False
):
    graph = get_base_graph(probes, links, directed, symmetrize)
    if should_include_latencies:
        latencies = [
            ((source_id, target_id), data['rtt'])
            for source_id, target_id, data in graph.edges(data=True)
        ]
    if should_remove_tivs:
        graph = remove_tivs(graph)
    if epsilon is not None:
        graph = threshold_graph(graph, epsilon)
    if clustering_distance is not None:
        graph = cluster_graph(graph, clustering_distance)
    if should_compute_curvatures:
        graph = compute_ricci_curvatures(graph, ricci_curvature_alpha, ricci_curvature_weight_label, routes, traffic_matrix, force_optimal_transport)
    if should_include_latencies:
        return graph, latencies
    else:
        return graph

def get_graph_from_csvs(
    path_probes,
    path_links,
    *,
    epsilon=None,
    clustering_distance=None,
    should_remove_tivs=False,
    should_include_latencies=False,
    should_compute_curvatures=True,
    ricci_curvature_alpha=0.,
    ricci_curvature_weight_label=None,
    throughputs_for_curvature=False,
    directed=False,
    symmetrize=False,
    force_optimal_transport=False  # Unused, but kept for consistent interface
):
    """
    Generate a NetworkX graph and optionally a list of latencies.

    As input, take two CSV files (for nodes and edges), and a special
    cutoff parameter `epsilon` that determines when an edge should be
    included in the graph.

    Additionally take a parameter for the clustering algorithm. The
    parameter is treated in meters (think roughly how closely two cities
    should be in order to be in the same cluster).

    Also take in a flag determining whether triangle inequality
    violations (TIVs) should be removed. This simply removes the edges
    that contribute to a TIV in a conservative fashion (i.e., the edges
    that are too long for any triangle in the graph).

    Finally, take in a flag indicating whether to return latencies in
    the format [((source_id, target_id), rtt), ...].
    """
    with open(path_probes) as file_probes, open(path_links) as file_links:
        probes = csv.DictReader(file_probes)
        links = csv.DictReader(file_links)

        routes = None
        traffic_matrix = None
        # if throughputs_for_curvature:
        #     routes = tomography.get_shortest_routes()

        return get_graph(
            probes, links,
            epsilon=epsilon,
            clustering_distance=clustering_distance,
            should_remove_tivs=should_remove_tivs,
            should_include_latencies=should_include_latencies,
            should_compute_curvatures=should_compute_curvatures,
            ricci_curvature_alpha=ricci_curvature_alpha,
            ricci_curvature_weight_label=ricci_curvature_weight_label,
            directed=directed, symmetrize=symmetrize,
            routes=routes, traffic_matrix=traffic_matrix, force_optimal_transport=force_optimal_transport
        )

def get_graph_from_json(
    path,
    *,
    epsilon=None,
    clustering_distance=None,
    should_remove_tivs=False,
    should_include_latencies=False,
    should_compute_curvatures=True,
    ricci_curvature_alpha=0.,
    ricci_curvature_weight_label=None,
    directed=False,
    symmetrize=False,
    force_optimal_transport=False,
    return_traffic_matrix=False,
):
    with open(path) as file:
        blob = json.load(file)

        probes = blob['nodes']
        links = {
            (link['source_id'], link['target_id']): link
            for link in blob['links']
        }
        routes = None
        traffic_matrix = None
        # TODO: Allow multiple routes to have the same source and destination
        if 'routes' in blob:
            routes = {}
            traffic_matrix = {}
            for link in links.values():
                link['throughput'] = 0.
            for route_info in blob['routes']:
                route = route_info['route']
                volume = route_info['volume']
                for u, v in itertools.pairwise(route):
                    links[(u, v)]['throughput'] += volume
                if route[0] not in routes:
                    routes[route[0]] = {}
                routes[route[0]][route[-1]] = route
                traffic_matrix[route[0], route[-1]] = volume

    graph = get_graph(
        probes, links.values(),
        epsilon=epsilon,
        clustering_distance=clustering_distance,
        should_remove_tivs=should_remove_tivs,
        should_include_latencies=should_include_latencies,
        should_compute_curvatures=should_compute_curvatures,
        ricci_curvature_alpha=ricci_curvature_alpha,
        ricci_curvature_weight_label=ricci_curvature_weight_label,
        directed=directed,
        symmetrize=symmetrize,
        routes=routes,
        traffic_matrix=traffic_matrix,
        force_optimal_transport=force_optimal_transport,
    )

    if return_traffic_matrix:
        return graph, traffic_matrix
    else:
        return graph

def get_network_data(
    graph: nx.Graph
):
    """
    Extract data from a NetworkX graph into dictionaries.

    The main use of this function is to turn the graph data that is
    indexed by node names into a format that is indexed by integers.

    Furthermore, this function will convert latitude-longitude positions
    into Cartesian coordinates using the Mercator projection. This is
    something that is subject to change, but is fine for now.

    The function returns:
    * A dictionary of "graph data," which contains geometry and
      connectivity information. This has three members: `coordinates`,
      `edges`, and `labels`, where coordinates is a list of x-y pairs,
      `edges` is a list of pairs of indices whose vertices are
      connected, and `labels` is the original names of the vertices in
      the NetworkX representation. Optionally, it includes a fourth
      `bounding_box` member describing the geographical extents of the
      network.
    * A dictionary of "vertex data," which maps attribute names to lists
      of their values on the nodes.
    * A dictionary of "edge data," which maps attribute names to lists
      of their values on the edges.
    """
    labels_to_indices = {
        node: index
        for index, node in enumerate(graph.nodes)
    }

    graph_data = {}
    graph_data['labels'] = list(graph.nodes)
    graph_data['coordinates'] = [
        utility.mercator(node['long'], node['lat'])
        for node in graph.nodes.values()
    ]
    graph_data['edges'] = [
        (
            labels_to_indices[source_label],
            labels_to_indices[target_label]
        )
        for source_label, target_label in graph.edges
    ]

    longitudes = [node['long'] for node in graph.nodes.values()]
    latitudes = [node['long'] for node in graph.nodes.values()]
    if 'long_min' in graph.graph:
        coordiate_min = utility.mercator(
            graph.graph['long_min'], graph.graph['lat_min']
        )
        coordiate_max = utility.mercator(
            graph.graph['long_max'], graph.graph['lat_max']
        )
        graph_data['bounding_box'] = (
            coordiate_min[0], coordiate_max[0],
            coordiate_min[1], coordiate_max[1]
        )
    else:
        # TODO: Implement this in a reasonable way
        graph_data['bounding_box'] = None

    vertex_data = {}
    vertex_attributes = set()
    for _, data in graph.nodes(data=True):
        for attribute_name in data:
            vertex_attributes.add(attribute_name)
    for attribute_name in vertex_attributes:
        vertex_data[attribute_name] = [
            data[attribute_name]
            if attribute_name in data
            else None
            for _, data in graph.nodes(data=True)
        ]

    edge_data = {}
    edge_attributes = set()
    for _, _, data in graph.edges(data=True):
        for attribute_name in data:
            edge_attributes.add(attribute_name)
    for attribute_name in edge_attributes:
        edge_data[attribute_name] = [
            data[attribute_name]
            if attribute_name in data
            else None
            for _, _, data in graph.edges(data=True)
        ]

    return graph_data, vertex_data, edge_data
