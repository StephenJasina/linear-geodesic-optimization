"""File containing utility go generate a graphml file."""

import argparse
import csv

from GraphRicciCurvature.OllivierRicci import OllivierRicci
import networkx as nx
import numpy as np

import cluster_probes
import investigate_TIVs
import utility


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

def get_base_graph(probes_filename, latencies_filename):
    # Create the graph
    graph = nx.Graph()

    # Create the RTT violation list
    rtt_violation_list = []

    # Get the vertecies
    lat_min = np.inf
    lat_max = -np.inf
    long_min = np.inf
    long_max = -np.inf
    with open(probes_filename) as probes_file:
        probes_reader = csv.DictReader(probes_file)
        for row in probes_reader:
            lat = float(row['latitude'])
            long = float(row['longitude'])
            graph.add_node(
                row['id'],
                city=row['city'], country=row['country'],
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
    with open(latencies_filename) as latencies_file:
        latencies_reader = csv.DictReader(latencies_file)
        for row in latencies_reader:
            id_source = row['source_id']
            id_target = row['target_id']
            lat_source = graph.nodes[id_source]['lat']
            long_source = graph.nodes[id_source]['long']
            lat_target = graph.nodes[id_target]['lat']
            long_target = graph.nodes[id_target]['long']
            rtt = row['rtt']
            if rtt is None or rtt == '':
                continue
            rtt = float(rtt)

            # Check how often the difference is larger than 0
            gcl = utility.get_GCL(
                [lat_source, long_source],
                [lat_target, long_target]
            )
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

def threshold_graph(graph, epsilon=np.inf):
    edges_to_delete = []
    for u, v, d in graph.edges(data=True):
        rtt = d['rtt']
        gcl = d['gcl']

        if rtt - gcl > epsilon:
            edges_to_delete.append((u, v))

    for u, v in edges_to_delete:
        graph.remove_edge(u, v)

    return graph

def cluster_graph(graph, distance, min_samples):
    circumference_earth = 40075016.68557849
    graph = cluster_probes.cluster_graph(
        graph,
        distance / circumference_earth,
        min_samples
    )
    return graph

def remove_tivs(graph):
    triangles = investigate_TIVs.compute_triangles(graph)
    goodnesses = investigate_TIVs.compute_goodnesses(triangles)
    tivs = [
        triangle
        for triangle, goodness in goodnesses.items()
        if goodness > 1.
    ]
    long_edges = investigate_TIVs.get_long_edges(graph, tivs)

    for u, v in long_edges:
        graph.remove_edge(u, v)

    return graph

def get_graph(
    probes_filename, latencies_filename, epsilon=np.inf,
    clustering_distance=None, clustering_min_samples=None,
    should_remove_tivs=False
):
    """
    Generate a NetworkX graph representing a delay space.

    As input, take two CSV files (for nodes and edges), and a special
    cutoff parameter `epsilon` that determines when an edge should be
    included in the graph.

    Additionally take the two parameters for the DBSCAN algorithm. The
    first parameter is treated in meters (think roughly how closely two
    cities should be in order to be in the same cluster).
    """
    graph = get_base_graph(probes_filename, latencies_filename)
    graph = threshold_graph(graph, epsilon)
    if clustering_distance is not None and clustering_min_samples is not None:
        graph = cluster_graph(graph, clustering_distance,
                              clustering_min_samples)
    if should_remove_tivs:
        graph = remove_tivs(graph)
    return graph

def compute_ricci_curvatures(graph):
    orc = OllivierRicci(graph, weight='weight', alpha=0.)
    graph = orc.compute_ricci_curvature()

    # Delete extraneous edge data
    for _, _, d in graph.edges(data=True):
        del d['weight']

    return graph

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
    parser.add_argument('--no-tivs', '-t', action='store_true',
                        dest='should_remove_tivs')
    parser.add_argument('--output', '-o', metavar='<filename>',
                        dest='output_filename', required=True)
    args = parser.parse_args()
    probes_filename = args.probes_filename
    latencies_filename = args.latencies_filename
    epsilon = args.epsilon
    if epsilon is None:
        epsilon = np.inf
    should_remove_tivs = args.should_remove_tivs
    output_filename = args.output_filename

    graph = get_graph(probes_filename, latencies_filename, epsilon,
                      500000, 2, should_remove_tivs)
    if should_remove_tivs:
        graph = remove_tivs(graph)
    graph = compute_ricci_curvatures(graph)
    nx.write_graphml(graph, f'{output_filename}')
