"""File containing utility go generate a graphml file."""

import argparse
import csv
import os
import sys

from GraphRicciCurvature.OllivierRicci import OllivierRicci
import networkx as nx
import numpy as np

def get_GCD_latency(latlong_a, latlong_b):
    radius_earth = 40075016.68557849 / (2 * np.pi)
    c = 299792458

    # Convert spherical coordinates to Cartesian coordinates
    latlong_a = np.array(latlong_a) * np.pi / 180
    p_a = np.array([
        np.cos(latlong_a[1]) * np.cos(latlong_a[0]),
        np.sin(latlong_a[1]) * np.cos(latlong_a[0]),
        np.sin(latlong_a[0])
    ])
    latlong_b = np.array(latlong_b) * np.pi / 180
    p_b = np.array([
        np.cos(latlong_b[1]) * np.cos(latlong_b[0]),
        np.sin(latlong_b[1]) * np.cos(latlong_b[0]),
        np.sin(latlong_b[0])
    ])

    # Note that the dot product is clamped to [-1, 1] to deal with
    # floating point error
    return np.arccos(min(max(p_a @ p_b, -1.), 1.)) \
        * radius_earth / (2. * c / 3.)

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

    # Get the edges
    latencies_reader = csv.DictReader(latencies_file)
    for row in latencies_reader:
        id_source = row['source_id']
        id_target = row['target_id']
        lat_source = graph.nodes[id_source]['lat']
        long_source = graph.nodes[id_source]['long']
        lat_target = graph.nodes[id_target]['lat']
        long_target = graph.nodes[id_target]['long']
        latency = float(row['rtt'])

        # Only add edges satisfying the cutoff requirement
        if (
            latency - get_GCD_latency(
                [lat_source, long_source],
                [lat_target, long_target]
            )
        ) < epsilon:
            graph.add_edge(id_source, id_target, weight=1.)

    # Compute the curvatures
    orc = OllivierRicci(graph, weight='weight', alpha=0.)
    graph = orc.compute_ricci_curvature()

    # Delete extraneous edge weights
    for _, _, d in graph.edges(data=True):
        del[d['weight']]

    return graph

if __name__ == '__main__':
    # Parse arugments
    parser = argparse.ArgumentParser()
    parser.add_argument('--latencies-file', '-l', metavar='<filename>',
                        dest='latencies_filename', required=True)
    parser.add_argument('--probes-file', '-p', metavar='<filename>',
                        dest='probes_filename', required=True)
    parser.add_argument('--epsilon', '-e', metavar='<epsilon>',
                        dest='epsilon', type=float, required=False)
    parser.add_argument('--output', '-o', metavar='<basename>',
                        dest='output_basename', required=True)
    args = parser.parse_args()
    latencies_filename = args.latencies_filename
    probes_filename = args.probes_filename
    epsilons = [args.epsilon]
    if args.epsilon is None:
        epsilons = list(range(1, 31))
    output_basename = args.output_basename

    for epsilon in epsilons:
        with open(probes_filename, 'r') as probes_file, \
                open(latencies_filename, 'r') as latencies_file:
            graph = get_graph(probes_file, latencies_file, epsilon)
            nx.write_graphml(graph, f'{output_basename}{epsilon}.graphml')
