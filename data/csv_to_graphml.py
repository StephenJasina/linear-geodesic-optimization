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
    return np.arccos(min(max(p_a @ p_b, -1.), 1.)) \
        * radius_earth / (2. * c / 3.)

if __name__ == '__main__':
    # Parse arugments
    parser = argparse.ArgumentParser()
    parser.add_argument('--latencies-file', '-l', metavar='<filename>',
                        dest='latenciesFilename', required=True)
    parser.add_argument('--coordinates-file', '-c', metavar='<filename>',
                        dest='coordinatesFilename', required=True)
    parser.add_argument('--epsilon', '-e', metavar='<epsilon>',
                        dest='epsilon', type=float, required=False)
    parser.add_argument('--output-file', '-o', metavar='<filename>',
                        dest='outputFilename', required=False)
    args = parser.parse_args()
    latenciesFilename = args.latenciesFilename
    coordinatesFilename = args.coordinatesFilename
    epsilon = args.epsilon
    if epsilon is None:
        epsilon = np.inf
    outputFilename = args.outputFilename

    # Create the graph
    graph = nx.Graph()

    with open(coordinatesFilename, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            graph.add_node(
                row['id'],
                city=row['city'], country=row['country'],
                lat=float(row['latitude']), long=float(row['longitude'])
            )

    with open(latenciesFilename, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            id_source = row['source_id']
            id_target = row['target_id']
            lat_source = graph.nodes[id_source]['lat']
            long_source = graph.nodes[id_source]['long']
            lat_target = graph.nodes[id_target]['lat']
            long_target = graph.nodes[id_target]['long']
            latency = float(row['rtt'])
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
    for u, v, d in graph.edges(data=True):
        del[d['weight']]

    # Write the graphml
    if outputFilename is None:
        for line in nx.generate_graphml(graph):
            print(line)
    else:
        nx.write_graphml(graph, outputFilename)
