import argparse
import itertools
import sys

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from linear_geodesic_optimization.data import input_network, utility
from linear_geodesic_optimization import plot


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graphml', '-g', dest='graphml_filename', metavar='graphml-file')
    parser.add_argument('--probes', '-p', dest='probes_filename', metavar='probes-file')
    parser.add_argument('--latencies', '-l', dest='latencies_filename', metavar='latencies-file')
    parser.add_argument('--epsilon', '-e', dest='epsilon', metavar='epsilon', type=float)
    parser.add_argument('--clustering-distance', '-c', dest='clustering_distance', metavar='clustering-distance', type=float)
    parser.add_argument('--show-map', '-m', dest='show_map', action='store_true')
    parser.add_argument('--output', '-o', dest='output_filename', metavar='filename')
    args = parser.parse_args()
    graphml_filename = args.graphml_filename
    probes_filename = args.probes_filename
    latencies_filename = args.latencies_filename
    epsilon = args.epsilon
    clustering_distance = args.clustering_distance
    show_map = args.show_map
    output_filename = args.output_filename

    if graphml_filename is not None:
        graph = nx.graphml.read_graphml(graphml_filename)
        # graph = input_network.cluster_graph(graph, 500000)
        # graph = input_network.compute_ricci_curvatures(graph)
        # graph = input_network.compute_curvatures_from_throughputs(graph)
        curvatures = [edge_data['ricciCurvature'] for _, _, edge_data in graph.edges(data=True)]
        print(min(curvatures), max(curvatures))
    else:
        if probes_filename is None or latencies_filename is None:
            print('Need to supply input files', file=sys.stderr)
            sys.exit(-1)

        graph = input_network.get_graph_from_paths(
            probes_filename, latencies_filename,
            epsilon=epsilon,
            ricci_curvature_alpha=0.,
            clustering_distance=clustering_distance,
        )

    coordinates = np.array([
        utility.mercator(data['long'], data['lat'])
        for _, data in graph.nodes(data = True)
    ])

    fig, ax = plt.subplots(1, 1, facecolor='#808080')

    if show_map:
        scale = 0.8
        image_data, extent = plot.get_image_data(coordinates, 1000, scale)
        ax.imshow(image_data, extent = extent)

    plot.get_network_plot(
        graph, ax = ax,
        # weight_label='throughput', color_min=-1., color_max=1.
    )

    if output_filename is None:
        plt.show()
    else:
        fig.savefig(output_filename, dpi=200, bbox_inches='tight')
