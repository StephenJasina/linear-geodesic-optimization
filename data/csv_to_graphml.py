"""File containing utility go generate a graphml file."""

import argparse
import csv
import os
import sys

from GraphRicciCurvature.OllivierRicci import OllivierRicci
import networkx as nx
import numpy as np

import cluster_probes
import utility

def get_graph(probes_file, latencies_file, epsilon):
    # Create the graph
    graph = nx.Graph()

    # Get the vertecies
    probes_reader = csv.DictReader(probes_file)
    for row in probes_reader:
        graph.add_node(
            row['id'],
            city=row['city'], country=row['country'],
            lat=float(row['latitude']), long=float(row['longitude'])
        )
        if 'cluster' in row:
            graph.nodes[row['id']]['cluster'] = int(row['cluster'])

    # Get the edges
    latencies_reader = csv.DictReader(latencies_file)
    for row in latencies_reader:
        id_source = row['source_id']
        id_target = row['target_id']
        lat_source = graph.nodes[id_source]['lat']
        long_source = graph.nodes[id_source]['long']
        lat_target = graph.nodes[id_target]['lat']
        long_target = graph.nodes[id_target]['long']
        rtt = float(row['rtt'])

        # Only add edges satisfying the cutoff requirement
        if (
            rtt - utility.get_GCD_latency(
                [lat_source, long_source],
                [lat_target, long_target]
            )
        ) < epsilon:
            # If there is multiple sets of rtt data for a single edge,
            # only pay attention to the minimal one
            if ((id_source, id_target) not in graph.edges
                    or graph.edges[id_source,id_target]['rtt'] > rtt):
                graph.add_edge(id_source, id_target, weight=1., rtt=rtt)

    # Simplify the graph by clustering
    distance = 300000
    eps = distance / 40075016.68557849
    graph = cluster_probes.cluster_graph(graph, eps, 2)

    # Compute the curvatures
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

    print(graph.number_of_nodes(), graph.number_of_edges())

    return graph

if __name__ == '__main__':
    # Parse arugments
    parser = argparse.ArgumentParser()
    parser.add_argument('--latencies-file', '-l', metavar='<filename>',
                        dest='latencies_filename', required=True)
    parser.add_argument('--probes-file', '-p', metavar='<filename>',
                        dest='probes_filename', required=True)
    parser.add_argument('--epsilon', '-e', metavar='<epsilon>',
                        dest='epsilon', type=int, required=False)
    parser.add_argument('--output', '-o', metavar='<basename>',
                        dest='output_basename', required=True)
    args = parser.parse_args()
    latencies_filename = args.latencies_filename
    probes_filename = args.probes_filename
    epsilons = [args.epsilon]
    if args.epsilon is None:
        epsilons = list(range(1, 21))
    output_basename = args.output_basename

    for epsilon in epsilons:
        with open(probes_filename, 'r') as probes_file, \
                open(latencies_filename, 'r') as latencies_file:
            graph = get_graph(probes_file, latencies_file, epsilon)
            nx.write_graphml(graph, f'{output_basename}{epsilon}.graphml')
