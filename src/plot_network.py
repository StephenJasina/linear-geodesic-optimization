import argparse

from matplotlib import pyplot as plt
import networkx as nx

from linear_geodesic_optimization.plot import get_network_plot


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_filename', metavar='graphml-file')
    parser.add_argument('--output', '-o', metavar='filename',
                        dest='output_filename')
    args = parser.parse_args()
    input_filename = args.input_filename
    output_filename = args.output_filename

    network_plot = get_network_plot(nx.graphml.read_graphml(input_filename))
    if output_filename is not None:
        network_plot.savefig('network.png', dpi=1000, bbox_inches='tight')
    plt.show()
