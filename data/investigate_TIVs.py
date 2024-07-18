import argparse
import csv
import collections
import os
import sys
import typing

import matplotlib.pyplot as plt
import networkx as nx

sys.path.append(os.path.join('..', 'src'))
from linear_geodesic_optimization.data \
    import triangle_inequality_violations as tivs
from linear_geodesic_optimization.data import input_network


def plot_goodnesses(goodnesses):
    fig = plt.figure()
    ax_1, ax_2 = fig.subplots(1, 2, sharey=True)
    if goodnesses:
        ax_1.ecdf(goodnesses.values())
    ax_1.plot([1, 1], [0, 1], 'r-.')
    ax_1.set_title('Goodness (All Triangles)')
    ax_1.set_xlabel('Goodness (lower = better)')
    ax_1.set_ylabel('CDF')

    goodness_violators = [goodness for goodness in goodnesses.values() if goodness > 1]
    if goodness_violators:
        ax_2.ecdf(goodness_violators)
    ax_2.set_title('Goodness (Violators Only)')
    ax_2.set_xlabel('Goodness (lower = better)')
    ax_2.set_ylabel('CDF')
    plt.show()

if __name__ == '__main__':
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
    args = parser.parse_args()
    probes_filename = args.probes_filename
    latencies_filename = args.latencies_filename
    epsilon = args.epsilon
    if epsilon is None:
        epsilon = float('inf')

    graph = input_network.get_graph_from_paths(
        probes_filename, latencies_filename, epsilon, 500000
    )

    triangles = tivs.compute_triangles(graph)
    goodnesses = tivs.compute_goodnesses(triangles, use_r=False)

    violating_triangles = [
        triangle
        for triangle, goodness in goodnesses.items()
        if goodness > 1.
    ]

    long_edges = tivs.get_long_edges(graph, violating_triangles)

    tiv_edge_sets = collections.defaultdict(set)
    for i, (u, v, w) in enumerate(violating_triangles):
        tiv_edge_sets[u,v].add(i)
        tiv_edge_sets[u,w].add(i)
        tiv_edge_sets[v,w].add(i)
    tiv_edge_set_cover = tivs.compute_greedy_set_cover(tiv_edge_sets)

    tiv_vertex_sets = collections.defaultdict(set)
    for i, (u, v, w) in enumerate(violating_triangles):
        tiv_vertex_sets[u].add(i)
        tiv_vertex_sets[v].add(i)
        tiv_vertex_sets[w].add(i)
    tiv_vertex_set_cover = tivs.compute_greedy_set_cover(tiv_vertex_sets)

    print(f'Proportion of TIVs (triangles): {(len(violating_triangles) / len(goodnesses)):.4f}')
    print(f'Proportion of long edges: {len(long_edges) / len(graph.edges):.4f}')
    print('Approximate proportion of TIVs (edges): '
          + f'{len(tiv_edge_set_cover) / len(graph.edges):.4f}')
    print('Approximate proportion of TIVs (vertices): '
          + f'{len(tiv_vertex_set_cover) / len(graph.nodes):.4f}')
    print(f'Worst goodness: {max(goodnesses.values()):.4f}')

    plot_goodnesses(goodnesses)

    thresholds = list(sorted(
        threshold
        for _, _, d in graph.edges(data=True)
        for threshold in (d['rtt'] - d['gcl'],)
    ))
    edge_counts = []
    long_edge_counts = []
    for i in range(0, len(thresholds), len(thresholds) // 100):
        threshold = thresholds[i]
        graph = input_network.get_graph_from_paths(
            probes_filename, latencies_filename, threshold, 500000
        )

        triangles = tivs.compute_triangles(graph)
        goodnesses = tivs.compute_goodnesses(triangles, use_r=False)

        violating_triangles = [
            triangle
            for triangle, goodness in goodnesses.items()
            if goodness > 1.05
        ]

        long_edges = tivs.get_long_edges(graph, violating_triangles)

        edge_counts.append(len(graph.edges))
        long_edge_counts.append(len(long_edges))
        print(len(graph.edges), len(long_edges), len(graph.edges) / max(1, len(long_edges)))

    plt.plot(edge_counts, long_edge_counts)
    plt.xlabel('Number of Total Edges')
    plt.ylabel('Number of Violating Edges')
    plt.title('Proportion of Violating Edges across Thresholds')
    plt.show()
