import argparse
import os
import pickle

from linear_geodesic_optimization import data

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

    data_file_path = os.path.join('..', 'data', parameters['data_file_name'])
    latency_file_name = parameters['latency_file_name']
    latency_file_path = None
    if latency_file_name is not None:
        os.path.join('..', 'data', latency_file_name)
    network = data.read_graphml(data_file_path, latency_file_path)

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