"""File containing utility go generate a graphml file."""

import argparse
import csv

from GraphRicciCurvature.OllivierRicci import OllivierRicci
import networkx as nx
import numpy as np

import cluster_probes
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

def get_graph(
    probes_filename, latencies_filename, epsilon,
    clustering_distance=None, clustering_min_samples=None
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
    # Create the graph
    graph = nx.Graph()

    # Create the RTT violation list
    rtt_violation_list = []

    # Get the vertecies
    with open(probes_filename) as probes_file:
        probes_reader = csv.DictReader(probes_file)
        for row in probes_reader:
            graph.add_node(
                row['id'],
                city=row['city'], country=row['country'],
                lat=float(row['latitude']), long=float(row['longitude'])
            )

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
            gcd_latency = utility.get_GCD_latency(
                [lat_source, long_source],
                [lat_target, long_target]
            )
            if rtt - gcd_latency < 0:
                rtt_violation_list.append((id_source, id_target))
                print(id_source, id_target, graph.nodes[id_source], graph.nodes[id_target], rtt, gcd_latency)

            # Only add edges satisfying the cutoff requirement
            if (
                rtt - utility.get_GCD_latency(
                    [lat_source, long_source],
                    [lat_target, long_target]
                )
            ) < epsilon:
                # If there is multiple sets of RTT data for a single
                # edge, only pay attention to the minimal one
                if ((id_source, id_target) not in graph.edges
                        or graph.edges[id_source,id_target]['rtt'] > rtt):
                    graph.add_edge(id_source, id_target, weight=1., rtt=rtt)

    # Delete nodes with inconsistent geolocation
    nodes_to_delete = minimize_id_removal(rtt_violation_list)

    for node in nodes_to_delete:
        graph.remove_node(node)

    # Simplify the graph by clustering
    if clustering_distance is not None and clustering_min_samples is not None:
        circumference_earth = 40075016.68557849
        graph = cluster_probes.cluster_graph(graph,
                              clustering_distance / circumference_earth,
                              clustering_min_samples)

    # Compute the curvatures. This adds attributes called
    # `ricciCurvature` to the vertices and edges of the graph
    orc = OllivierRicci(graph, weight='weight', alpha=0.)
    graph = orc.compute_ricci_curvature()

    # Delete extraneous edge data
    for _, _, d in graph.edges(data=True):
        del d['weight']
        del d['rtt']

    # Delete nodes with no edges
    nodes = list(graph.nodes)
    for node in nodes:
        if len(graph.edges(node)) == 0:
            graph.remove_node(node)

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
    parser.add_argument('--ip-type', '-i', type=str, required=True,
                        dest='ip_type', metavar='<ipv4/ipv6>',
                        help='Type of IP (e.g., ipv4, ipv6).')
    parser.add_argument('--epsilon', '-e', type=int, required=False,
                        dest='epsilon', metavar='<epsilon>',
                        help='Residual threshold')
    parser.add_argument('--output', '-o', metavar='<basename>',
                        dest='output_filename', required=True)
    args = parser.parse_args()
    latencies_filename = args.latencies_filename
    ip_type = args.ip_type
    probes_filename = args.probes_filename
    epsilons = [args.epsilon]
    if args.epsilon is None:
        epsilons = list(range(2, 40))
    output_filename = args.output_filename

    for epsilon in epsilons:
        graph = get_graph(probes_filename, latencies_filename, epsilon,
                          300000, 2)
        nx.write_graphml(graph, f'output_filename')
