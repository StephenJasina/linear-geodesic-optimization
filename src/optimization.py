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

import networkx as nx
import numpy as np
import scipy

from linear_geodesic_optimization.data import input_network
from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh
from linear_geodesic_optimization.optimization import optimization


# Error on things like division by 0
warnings.simplefilter('error')

def main(probes_filename, latencies_filename, epsilon, clustering_distance,
         lambda_curvature, lambda_smooth, lambda_geodesic,
         initial_radius, sides, mesh_scale,
         leaveout_proportion=0.,
         maxiter=1000, output_dir_name=os.path.join('..', 'out'),
         graphml_filename=None,
         initialization_file_path=None):
    # Construct a mesh
    width = height = sides
    mesh = RectangleMesh(width, height, mesh_scale)

    # Construct the network graph
    if graphml_filename is None:
        probes_file_path = os.path.join('..', 'data', probes_filename)
        latencies_file_path = os.path.join('..', 'data', latencies_filename)
        network, latencies = input_network.get_graph(
            probes_file_path, latencies_file_path,
            epsilon, clustering_distance,
            should_include_latencies=True
        )
    else:
        graphml_file_path = os.path.join('..', 'data', graphml_filename)
        network = nx.read_graphml(graphml_file_path)
        latencies = []
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
        f'{lambda_curvature}_{lambda_smooth}_{lambda_geodesic}_{initial_radius}_{width}_{height}_{mesh_scale}'
    )
    if os.path.isdir(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

    parameters = {
        'epsilon': epsilon,
        'clustering_distance': clustering_distance,
        'should_remove_TIVs': False, # TODO: Pass this as a parameter?
        'lambda_curvature': lambda_curvature,
        'lambda_smooth': lambda_smooth,
        'lambda_geodesic': lambda_geodesic,
        'initial_radius': initial_radius,
        'width': width,
        'height': height,
        'mesh_scale': mesh_scale,
        'coordinates_scale': 0.8,
        'leaveout_count': leaveout_count,
        'leaveout_seed': leaveout_seed
    }
    if graphml_filename is None:
        parameters['probes_filename'] = probes_filename
        parameters['latencies_filename'] = latencies_filename
    else:
        parameters['graphml_filename'] = graphml_filename

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
        network_latencies, 1.01 * 2**0.5 * mesh_scale / width,
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
    # minimizer_kwargs['options']['maxiter'] = 0
    # z = scipy.optimize.basinhopping(
    #     f, x0=z_0,
    #     niter=100, T=0.01,
    #     callback=computer.diagnostics,
    #     # take_step=mesh.take_step
    # ).x
    with open(os.path.join(directory, 'output'), 'wb') as f:
        pickle.dump({
            'parameters': parameters,
            'initial': optimization.Computer.to_float_list(z_0),
            'final': optimization.Computer.to_float_list(z),
            'network': network, # TODO: JSONify this
        }, f)

if __name__ == '__main__':
    # graphml_filenames = [
    #     os.path.join('toy', 'elbow.graphml')
    # ]
    data_dir = os.path.join('..', 'data')
    input_dir = os.path.join('ipv4', 'graph_US_IEEE')
    probes_filenames = [
        os.path.join(input_dir, filename)
        for filename in os.listdir(os.path.join(data_dir, input_dir))
        if filename.startswith('probes')
    ]
    latencies_filenames = [
        os.path.join(input_dir, 'latencies' + os.path.basename(probes_filename)[6:])
        for probes_filename in probes_filenames
    ]

    count = len(probes_filenames)

    epsilon = 10
    epsilons = [epsilon] * count
    clustering_distances = [500000] * count

    lambda_curvatures = [1.] * count
    lambda_smooths = [0.002] * count
    lambda_geodesics = [0.] * count
    initial_radii = [20.] * count
    sides = [50] * count
    mesh_scales = [1.] * count

    leaveout_proportions = [1.] * count

    max_iters = [100000] * count

    output_dir_names = [
        os.path.join('..', f'out_US_IEEE_{epsilon}', os.path.basename(probes_filename)[6:])
        for probes_filename in probes_filenames
    ]

    arguments = list(zip(
        probes_filenames, latencies_filenames, epsilons, clustering_distances,
        lambda_curvatures, lambda_smooths, lambda_geodesics,
        initial_radii, sides, mesh_scales,
        leaveout_proportions,
        max_iters,
        output_dir_names,
        # graphml_filenames,
    ))
    # Need to use ProcessPoolExecutor instead of multiprocessing.Pool
    # to allow child processes to spawn their own subprocesses
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for _ in executor.map(main, *zip(*arguments)):
            pass
