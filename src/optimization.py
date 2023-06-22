import datetime
import itertools
import json
import multiprocessing
import os
import pickle
import shutil
import time
import warnings

import numpy as np
import scipy

from linear_geodesic_optimization import data
from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh
from linear_geodesic_optimization.optimization import optimization


# Error on things like division by 0
warnings.simplefilter('error')

def main(data_file_name, latency_file_name,
         lambda_curvature, lambda_smooth, lambda_geodesic,
         initial_radius, sides, scale,
         leaveout_proportion=0.,
         maxiter=1000, output_dir_name=os.path.join('..', 'out')):
    data_file_path = os.path.join('..', 'data', data_file_name)
    data_name, _ = os.path.splitext(os.path.basename(data_file_name))
    latency_file_path = os.path.join('..', 'data', latency_file_name) \
        if latency_file_name is not None \
        else None

    # Construct a mesh
    width = height = sides
    mesh = RectangleMesh(width, height, scale)

    network_coordinates, network_edges, network_curvatures, network_latencies \
        = data.read_graphml(data_file_path, latency_file_path)
    network_vertices = mesh.map_coordinates_to_support(np.array(network_coordinates), np.float64(0.8))
    leaveout_count = int(leaveout_proportion * len(network_latencies))
    leaveout_seed = time.monotonic_ns() % (2**31 - 1)
    if leaveout_count > 0:
        rng = np.random.default_rng(leaveout_seed)
        rng.shuffle(network_latencies)
        network_latencies = network_latencies[:-leaveout_count]

    # Setup snapshots
    directory = os.path.join(
        output_dir_name, f'{data_name}',
        f'{lambda_curvature}_{lambda_smooth}_{lambda_geodesic}_{initial_radius}_{width}_{height}_{scale}'
    )
    if os.path.isdir(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

    with open(os.path.join(directory, 'parameters'), 'wb') as f:
        pickle.dump({
            'data_file_name': data_file_name,
            'latency_file_name': latency_file_name,
            'lambda_curvature': lambda_curvature,
            'lambda_smooth': lambda_smooth,
            'lambda_geodesic': lambda_geodesic,
            'initial_radius': initial_radius,
            'width': width,
            'height': height,
            'scale': scale,
            'leaveout_count': leaveout_count,
            'leaveout_seed': leaveout_seed
        }, f)

    # Initialize mesh
    z = np.array([
        (initial_radius**2
            - (i / (width - 1) - 0.5)**2
            - (j / (height - 1) - 0.5)**2)**0.5
        for i in range(width)
        for j in range(height)
    ]).reshape((width * height,))
    z = z - np.amin(z)
    # with open('initialization', 'rb') as f:
    #     z = np.array(pickle.load(f)['mesh_parameters'])
    z = mesh.set_parameters(z)

    computer = optimization.Computer(
        mesh, network_vertices, network_edges, network_curvatures,
        network_latencies, 1.01 * 2**0.5 * scale / width,
        lambda_curvature, lambda_smooth, lambda_geodesic,
        directory)

    f = computer.forward
    g = computer.reverse

    computer.diagnostics(None)
    scipy.optimize.minimize(f, z, method='L-BFGS-B', jac=g,
                            callback=computer.diagnostics,
                            options={'maxiter': maxiter})

if __name__ == '__main__':
    data_file_names = [
        os.path.join('graph_US', f'graph{i}.graphml')
        # for i in [4, 10, 12, 14, 16, 18, 22]
        for i in [16]
    ]
    latency_file_names = ['latencies_US.csv']
    lambda_curvatures = [1.]
    lambda_smooths = [0.0002]
    lambda_geodesics = [0., 0.0001, 0.0002, 0.0004, 0.001, 0.002, 0.004, 0.01, 0.02, 0.04, 0.1, 0.2, 0.4, 1., 2., 4., 10., 20., 40., 100., 200., 400., 1000., 2000., 4000., 10000., 20000., 40000., 100000., 200000., 400000.]
    initial_radii = [20.]
    sides = [40]
    scales = [1.]
    leaveout_proportions = [0.]

    arguments = list(itertools.product(
        data_file_names, latency_file_names,
        lambda_curvatures, lambda_smooths, lambda_geodesics,
        initial_radii, sides, scales,
        leaveout_proportions,
        [1000],
        [os.path.join('..', 'out_paper')]
    ))
    with multiprocessing.Pool() as p:
        p.starmap(main, arguments)
    # main(*arguments[0])
