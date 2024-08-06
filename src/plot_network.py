import argparse
import sys

from matplotlib import pyplot as plt
import networkx as nx

from linear_geodesic_optimization.data import input_network
from linear_geodesic_optimization.plot import get_network_plot


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graphml', '-g', dest='graphml_filename', metavar='graphml-file')
    parser.add_argument('--probes', '-p', dest='probes_filename', metavar='probes-file')
    parser.add_argument('--latencies', '-l', dest='latencies_filename', metavar='latencies-file')
    parser.add_argument('--epsilon', '-e', dest='epsilon', metavar='epsilon', type=float)
    parser.add_argument('--output', '-o', dest='output_filename', metavar='filename')
    args = parser.parse_args()
    graphml_filename = args.graphml_filename
    probes_filename = args.probes_filename
    latencies_filename = args.latencies_filename
    epsilon = args.epsilon
    output_filename = args.output_filename

    if graphml_filename is not None:
        graph = nx.graphml.read_graphml(graphml_filename)
    else:
        if probes_filename is None or latencies_filename is None:
            print('Need to supply input files', file=sys.stderr)
            sys.exit(-1)

        graph = input_network.get_graph_from_paths(probes_filename, latencies_filename, epsilon, ricci_curvature_alpha=0.9999)

    network_plot = get_network_plot(graph, weight_label='ricciCurvature')

    if output_filename is None:
        plt.show()
        curvatures = [
            d['ricciCurvature']
            for _, _, d in graph.edges(data=True)
        ]
        plt.hist(curvatures, bins=100)
        plt.show()
    else:
        network_plot.savefig(output_filename, dpi=1000, bbox_inches='tight')
