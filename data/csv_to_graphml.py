"""File containing utility to generate a graphml file."""

import argparse
import csv
import os
import sys

import networkx as nx
import numpy as np

sys.path.append(os.path.join('..', 'src'))
from linear_geodesic_optimization.data import input_network


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
    parser.add_argument('--clustering-distance', '-c', type=float,
                        required=False, dest='clustering_distance',
                        metavar='<clustering distance>')
    parser.add_argument('--throughputs-for-curvature', '-t',
                        action='store_true', dest='throughputs_for_curvature')
    parser.add_argument('--output', '-o', metavar='<filename>',
                        dest='output_filename', required=True)
    args = parser.parse_args()
    probes_filename = args.probes_filename
    latencies_filename = args.latencies_filename
    epsilon = args.epsilon
    if epsilon is None:
        epsilon = np.inf
    clustering_distance = args.clustering_distance
    throughputs_for_curvature = args.throughputs_for_curvature
    output_filename = args.output_filename

    graph = input_network.get_graph_from_paths(
        probes_filename, latencies_filename,
        epsilon=epsilon,
        clustering_distance=clustering_distance,
        throughputs_for_curvature=throughputs_for_curvature
    )
    nx.write_graphml(graph, f'{output_filename}')
