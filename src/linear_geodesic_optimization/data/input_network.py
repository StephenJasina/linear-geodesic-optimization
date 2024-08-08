import csv
import typing

from GraphRicciCurvature.OllivierRicci import OllivierRicci
import networkx as nx
import numpy as np

from linear_geodesic_optimization.data import clustering
from linear_geodesic_optimization.data \
    import triangle_inequality_violations as tivs
from linear_geodesic_optimization.data import utility


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

def get_base_graph(probes, latencies):
    # Create the graph
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
            city=probe['city'], country=probe['country'],
            lat=lat, long=long
        )
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
    for latency in latencies:
        id_source = latency['source_id']
        id_target = latency['target_id']

        if id_source not in graph.nodes or id_target not in graph.nodes:
            # Skip maesurements we're missing the nodes for
            continue

        lat_source = graph.nodes[id_source]['lat']
        long_source = graph.nodes[id_source]['long']
        lat_target = graph.nodes[id_target]['lat']
        long_target = graph.nodes[id_target]['long']
        gcl = utility.get_GCL(
            [lat_source, long_source],
            [lat_target, long_target]
        )

        if 'rtt' in latency:
            rtt = latency['rtt']
            # Skip edges with no measured RTT
            if rtt == '':
                continue
            rtt = float(rtt)
        else:
            # If rtt isn't available at all, just add every edge (i.e.,
            # assume the input contains connectivity information)
            rtt = gcl

        # Check how often the difference is larger than 0
        # TODO: Should this be more lenient?
        if rtt - gcl < 0:
            rtt_violation_list.append((id_source, id_target))

        # If there is multiple sets of RTT data for a single
        # edge, only pay attention to the minimal one
        if ((id_source, id_target) not in graph.edges
                or graph.edges[id_source,id_target]['rtt'] > rtt):
            graph.add_edge(
                id_source, id_target,
                weight=1., rtt=rtt, gcl=gcl
            )

    # Delete nodes with inconsistent geolocation
    for node in minimize_id_removal(rtt_violation_list):
        graph.remove_node(node)

    return graph

def threshold_graph(graph, epsilon):
    edges_to_delete = []
    for u, v, d in graph.edges(data=True):
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

def compute_ricci_curvatures(graph, alpha=0.):
    orc = OllivierRicci(
        graph, weight='weight', alpha=alpha,
        method='OTD', proc=1
    )
    graph = orc.compute_ricci_curvature()

    # Delete extraneous edge data, and rescale Ricci curvature
    for _, _, d in graph.edges(data=True):
        del d['weight']
        d['ricciCurvature'] /= (1. - alpha)

    return graph

def get_graph(
    probes, latencies, epsilon=None,
    clustering_distance=None, should_remove_tivs=False,
    should_include_latencies=False,
    should_compute_curvatures=True,
    ricci_curvature_alpha=0.
):
    graph = get_base_graph(probes, latencies)
    if should_include_latencies:
        latencies = [
            ((source_id, target_id), data['rtt'])
            for source_id, target_id, data in graph.edges(data=True)
        ]
    if should_remove_tivs:
        graph = remove_tivs(graph)
    if clustering_distance is not None:
        graph = cluster_graph(graph, clustering_distance)
    if epsilon is not None:
        graph = threshold_graph(graph, epsilon)
    if should_compute_curvatures:
        graph = compute_ricci_curvatures(graph, ricci_curvature_alpha)
    if should_include_latencies:
        return graph, latencies
    else:
        return graph

def get_graph_from_paths(
    probes_file_path, latencies_file_path, epsilon=None,
    clustering_distance=None, should_remove_tivs=False,
    should_include_latencies=False,
    should_compute_curvatures=True,
    ricci_curvature_alpha=0.
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
    with open(probes_file_path) as probes_file, open(latencies_file_path) as latencies_file:
        probes = csv.DictReader(probes_file)
        latencies = csv.DictReader(latencies_file)
        return get_graph(
            probes, latencies, epsilon,
            clustering_distance, should_remove_tivs,
            should_include_latencies,
            should_compute_curvatures,
            ricci_curvature_alpha
        )

def extract_from_graph(
    graph: nx.Graph,
    latencies: typing.Optional[typing.Tuple[typing.Tuple[str, str],
                                            float]] = None,
    with_labels: bool = False
) -> typing.Union[
    typing.Tuple[
        typing.List[typing.Tuple[np.float64, np.float64]],
        typing.Optional[typing.Tuple[np.float64, np.float64,
                                     np.float64, np.float64]],
        typing.List[typing.Tuple[int, int]],
        typing.List[np.float64],
        typing.List[typing.Tuple[typing.Tuple[int, int], np.float64]]
    ],
    typing.Tuple[
        typing.List[typing.Tuple[np.float64, np.float64]],
        typing.Optional[typing.Tuple[np.float64, np.float64,
                                     np.float64, np.float64]],
        typing.List[typing.Tuple[int, int]],
        typing.List[np.float64],
        typing.List[typing.Tuple[typing.Tuple[int, int], np.float64]],
        typing.List[str]
    ]
]:
    """
    Extract a tuple of location and latency data from a graph.

    Given a graph, a list of latencies, and an optional boolean, return:
    * A list of (x, y) coordinate pairs representing vertices
    * A bounding box aruond the coordinates
    * A list of pairs of indices into the vertex list representing edges
    * A list of (Ollivier-Ricci) curvatures of each of the edges
    * A list of list of pairs representing measured latencies. This is
      stored in the format ((i_index, j_index), latency) (which is
      different from the ((source_id, target_id), latency) format of the
      input parameter)
    * Optionally, a list of labels for each of the vertices
    """
    coordinates: typing.List[typing.Tuple[np.float64, np.float64]] \
        = [utility.mercator(node['long'], node['lat'])
           for node in graph.nodes.values()]
    bounding_box = None
    if 'long_min' in graph.graph:
        coordiate_min = utility.mercator(graph.graph['long_min'],
                                 graph.graph['lat_min'])
        coordiate_max = utility.mercator(graph.graph['long_max'],
                                 graph.graph['lat_max'])
        bounding_box = (
            coordiate_min[0], coordiate_max[0],
            coordiate_min[1], coordiate_max[1]
        )
    label_to_index = {label: index
                      for index, label in enumerate(graph.nodes)}
    network_edges: typing.List[typing.Tuple[int, int]] \
        = [(label_to_index[u], label_to_index[v])
           for u, v in graph.edges]
    network_curvatures: typing.List[np.float64] \
        = [edge['ricciCurvature']
           for edge in graph.edges.values()]
    network_latencies: typing.List[typing.Tuple[typing.Tuple[int, int],
                                                np.float64]] = []

    if latencies is None:
        latencies = [
            ((source_id, target_id), data['rtt'])
            for source_id, target_id, data in graph.edges(data=True)
        ]

    for (source_id, target_id), rtt in latencies:
        if source_id in label_to_index and target_id in label_to_index:
            network_latencies.append((
                (label_to_index[source_id],
                 label_to_index[target_id]),
                np.float64(rtt)
            ))

    if with_labels:
        network_city: typing.List[str] = [
            node['city']
            for node in graph.nodes.values()
        ]

        return coordinates, bounding_box, network_edges, network_curvatures, \
            network_latencies, list(graph.nodes), network_city
    else:
        return coordinates, bounding_box, network_edges, network_curvatures, \
            network_latencies
