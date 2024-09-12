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
    parser.add_argument('--show-map', '-m', dest='show_map', action='store_true')
    parser.add_argument('--output', '-o', dest='output_filename', metavar='filename')
    args = parser.parse_args()
    graphml_filename = args.graphml_filename
    probes_filename = args.probes_filename
    latencies_filename = args.latencies_filename
    epsilon = args.epsilon
    show_map = args.show_map
    output_filename = args.output_filename

    if graphml_filename is not None:
        graph = nx.graphml.read_graphml(graphml_filename)
    else:
        if probes_filename is None or latencies_filename is None:
            print('Need to supply input files', file=sys.stderr)
            sys.exit(-1)

        graph = input_network.get_graph_from_paths(
            probes_filename, latencies_filename,
            epsilon, ricci_curvature_alpha=0.9999, clustering_distance=500000
        )

    coordinates = np.array([
        utility.mercator(data['long'], data['lat'])
        for _, data in graph.nodes(data = True)
    ])

    fig, ax = plt.subplots()

    if show_map:
        scale = 0.8
        image_data, extent = plot.get_image_data(coordinates, 1000, scale)
        ax.imshow(image_data, extent = extent)

    plot.get_network_plot(
        graph, ax = ax
    )

    if output_filename is None:
        plt.show()
    else:
        fig.savefig(output_filename, dpi=1000, bbox_inches='tight')
