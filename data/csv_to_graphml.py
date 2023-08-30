"""File containing utility go generate a graphml file."""

import argparse
import csv
import os
import sys

from GraphRicciCurvature.OllivierRicci import OllivierRicci
import networkx as nx

if __name__ == '__main__':
    # Parse arugments
    parser = argparse.ArgumentParser()
    parser.add_argument('--latencies-file', '-l', metavar='<filename>',
                        dest='latenciesFilename', required=True)
    parser.add_argument('--coordinates-file', '-c', metavar='<filename>',
                        dest='coordinatesFilename', required=False)
    parser.add_argument('--output-file', '-o', metavar='<filename>',
                        dest='outputFilename', required=False)
    args = parser.parse_args()
    latenciesFilename = args.latenciesFilename
    coordinatesFilename = args.coordinatesFilename
    outputFilename = args.outputFilename

    # Create the graph
    graph = nx.Graph()

    if coordinatesFilename is not None:
        with open(coordinatesFilename, 'r') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                id = row['id']
                latitude = row['latitude']
                longitude = row['longitude']
                graph.add_node(id, lat=latitude, long=longitude)

    with open(latenciesFilename, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            source = row['source']
            destination = row['destination']
            latency = float(row['latencies'])
            graph.add_edge(source, destination, latency=latency)

    # Compute the curvatures
    orc = OllivierRicci(graph, weight='latency', alpha=0)
    graph = orc.compute_ricci_curvature()

    # Write the graphml
    if outputFilename is None:
        for line in nx.generate_graphml(graph):
            print(line)
    else:
        nx.write_graphml(graph, outputFilename)
