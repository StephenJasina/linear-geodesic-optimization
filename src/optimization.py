import concurrent.futures
import datetime
import itertools
import json
import os
# TODO: Convert to plain text
import pickle
import shutil
import time
import warnings

import numpy as np
import scipy

from linear_geodesic_optimization.data import input_network
from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh
from linear_geodesic_optimization.optimization import optimization


# Error on things like division by 0
warnings.simplefilter('error')

def main(probes_filename, latencies_filename, epsilon, clustering_distance,
         lambda_curvature, lambda_smooth, lambda_geodesic,
         initial_radius, sides, scale,
         leaveout_proportion=0.,
         maxiter=1000, output_dir_name=os.path.join('..', 'out'),
         initialization_file_path=None):
    # Construct the network graph
    probes_file_path = os.path.join('..', 'data', probes_filename)
    latencies_file_path = os.path.join('..', 'data', latencies_filename)

    # Construct a mesh
    width = height = sides
    mesh = RectangleMesh(width, height, scale)

    network, latencies = input_network.get_graph(
        probes_file_path, latencies_file_path,
        epsilon, clustering_distance,
        should_include_latencies=True
    )
    network = input_network.extract_from_graph(network, latencies)
    network_coordinates, bounding_box, network_edges, network_curvatures, network_latencies = network
    network_vertices = mesh.map_coordinates_to_support(np.array(network_coordinates), np.float64(0.8), bounding_box)
    leaveout_count = int(leaveout_proportion * len(network_latencies))
    leaveout_seed = time.monotonic_ns() % (2**31 - 1)
    if leaveout_count > 0:
        rng = np.random.default_rng(leaveout_seed)
        rng.shuffle(network_latencies)
        network_latencies = network_latencies[:-leaveout_count]

    # Setup snapshots
    directory = os.path.join(
        output_dir_name,
        f'{lambda_curvature}_{lambda_smooth}_{lambda_geodesic}_{initial_radius}_{width}_{height}_{scale}'
    )
    if os.path.isdir(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

    parameters = {
        'probes_filename': probes_filename,
        'latencies_filename': latencies_filename,
        'epsilon': epsilon,
        'clustering_distance': clustering_distance,
        'should_remove_TIVs': False, # TODO: Pass this as a parameter?
        'lambda_curvature': lambda_curvature,
        'lambda_smooth': lambda_smooth,
        'lambda_geodesic': lambda_geodesic,
        'initial_radius': initial_radius,
        'width': width,
        'height': height,
        'scale': scale,
        'leaveout_count': leaveout_count,
        'leaveout_seed': leaveout_seed
    }
    with open(os.path.join(directory, 'parameters'), 'wb') as f:
        pickle.dump(parameters, f)

    # Initialize mesh
    if initialization_file_path is None:
        z_0 = np.array([
            (initial_radius**2
                - (i / (width - 1) - 0.5)**2
                - (j / (height - 1) - 0.5)**2)**0.5
            for i in range(width)
            for j in range(height)
        ]).reshape((width * height,))
        z_0 = z_0 - np.amin(z_0)
    else:
        with open(initialization_file_path, 'rb') as f:
            z_0 = np.array(pickle.load(f)['mesh_parameters'])
    z_0 = mesh.set_parameters(z_0)

    computer = optimization.Computer(
        mesh, network_vertices, network_edges, network_curvatures,
        network_latencies, 1.01 * 2**0.5 * scale / width,
        lambda_curvature, lambda_smooth, lambda_geodesic,
        directory)

    f = computer.forward
    g = computer.reverse

    computer.diagnostics(None)
    minimizer_kwargs = {
        'method': 'L-BFGS-B',
        'jac': g,
        'callback': computer.diagnostics,
        'options': {'maxiter': maxiter},
    }
    z = scipy.optimize.minimize(f, z_0, **minimizer_kwargs).x
    # z = scipy.optimize.dual_annealing(f,
    #                                   scipy.optimize.Bounds(
    #                                       -4. * np.ones(z_0.shape),
    #                                       4. * np.ones(z_0.shape)
    #                                   ),
    #                                   visit = 1.1,
    #                                 #   minimizer_kwargs = minimizer_kwargs,
    #                                   no_local_search = True,
    #                                   callback=computer.diagnostics,
    #                                   x0 = z_0
    # ).x
    with open(os.path.join(directory, 'output'), 'wb') as f:
        pickle.dump({
            'parameters': parameters,
            'initial': optimization.Computer.to_float_list(z_0),
            'final': optimization.Computer.to_float_list(z),
            'network': network, # TODO: JSONify this
        }, f)

if __name__ == '__main__':
    probes_filenames = [
        os.path.join('toy', 'elbow_stretch_probes.csv')
    ]
    latency_filenames = [
        os.path.join('toy', 'elbow_stretch_latencies.csv')
    ]
    epsilons = [10]
    clustering_distances = [0]

    lambda_curvatures = [1.]
    lambda_smooths = [0.004]
    lambda_geodesics = [0.]
    initial_radii = [20.]
    sides = [50]
    scales = [1.]

    leaveout_proportions = [1.]

    arguments = list(itertools.product(
        probes_filenames, latency_filenames, epsilons, clustering_distances,
        lambda_curvatures, lambda_smooths, lambda_geodesics,
        initial_radii, sides, scales,
        leaveout_proportions,
        [5],
        [os.path.join('..', 'out_test')]
    ))
    # Need to use ProcessPoolExecutor instead of multiprocessing.Pool
    # to allow child processes to spawn their own subprocesses
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for _ in executor.map(main, *zip(*arguments)):
            pass
