import argparse
import os
import pickle

import networkx as nx

from linear_geodesic_optimization.data import input_network


def main(directory, max_iterations):
    with open(os.path.join(directory, 'parameters'), 'rb') as f:
        parameters = pickle.load(f)

    with open(os.path.join(directory, '0'), 'rb') as f:
        z_0 = pickle.load(f)['mesh_parameters']

    iteration = max(int(name)
                    for name in os.listdir(directory)
                    if name.isdigit())
    if max_iterations is not None:
        iteration = min(iteration, max_iterations)
    with open(os.path.join(directory, str(iteration)), 'rb') as f:
        z = pickle.load(f)['mesh_parameters']

    if 'graphml_filename' in parameters:
        network = nx.read_graphml(parameters['graphml_filename'])
        latencies = []
    else:
        probes_file_path = os.path.join('..', 'data', parameters['probes_filename'])
        latencies_filename = parameters['latencies_filenames']
        if not isinstance(latencies_filename, str):
            latencies_filename = latencies_filename[0]
        latencies_file_path = os.path.join('..', 'data', latencies_filename)
        epsilon = parameters['epsilon']
        clustering_distance = parameters['clustering_distance']
        should_remove_tivs = parameters['should_remove_TIVs']
        ricci_curvature_alpha = parameters['ricci_curvature_alpha']
        network, latencies = input_network.get_graph_from_paths(
            probes_file_path, latencies_file_path,
            epsilon, clustering_distance,
            should_remove_tivs=should_remove_tivs,
            should_include_latencies=True,
            ricci_curvature_alpha=ricci_curvature_alpha
        )
    network = input_network.extract_from_graph_old(network, latencies)

    with open(os.path.join(directory, 'output'), 'wb') as f:
        pickle.dump({
            'parameters': parameters,
            'initial': z_0,
            'final': z,
            'network': network,
        }, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate an output file from a directory of snapshots')
    parser.add_argument('--directory', '-d', type=str, required=True,
                        dest='directory', metavar='<path>',
                        help='The input directory.')
    parser.add_argument('--max-iterations', '-m', type=int, required=False,
                        dest='max_iterations', metavar='<iterations>',
                        help='The number of iterations to limit to.')
    args = parser.parse_args()
    directory = args.directory
    max_iterations = args.max_iterations

    main(directory, max_iterations)