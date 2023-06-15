import datetime
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

def main(data_file_name, lambda_curvature, lambda_smooth, lambda_geodesic,
         initial_radius, width, height,
         leaveout_proportion=0.,
         maxiter=1000, output_dir_name=os.path.join('..', 'out')):
    data_file_path = os.path.join('..', 'data', data_file_name)
    data_name, _ = os.path.splitext(os.path.basename(data_file_name))

    # Construct a mesh
    mesh = RectangleMesh(width, height)

    network_coordinates, network_edges, network_curvatures, network_latencies \
        = data.read_graphml(data_file_path,
                            os.path.join('..', 'data', 'latencies_US.csv'))
    network_vertices = mesh.map_coordinates_to_support(np.array(network_coordinates), np.float64(0.5))
    leaveout_count = int(leaveout_proportion * len(network_latencies))
    leaveout_seed = time.monotonic_ns() % (2**31 - 1)
    if leaveout_count > 0:
        rng = np.random.default_rng(leaveout_seed)
        rng.shuffle(network_latencies)
        network_latencies = network_latencies[:-leaveout_count]

    # Setup snapshots
    directory = os.path.join(
        output_dir_name, f'{data_name}',
        f'{lambda_curvature}_{lambda_smooth}_{lambda_geodesic}_{initial_radius}_{width}_{height}'
    )
    if os.path.isdir(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

    with open(os.path.join(directory, 'parameters'), 'wb') as f:
        pickle.dump({
            'data_file_name': data_file_name,
            'lambda_curvature': lambda_curvature,
            'lambda_smooth': lambda_smooth,
            'lambda_geodesic': lambda_geodesic,
            'initial_radius': initial_radius,
            'width': width,
            'height': height,
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
        network_latencies, 1.01 * 2**0.5 / width,
        lambda_curvature, lambda_smooth, lambda_geodesic,
        directory)

    f = computer.forward
    g = computer.reverse

    computer.diagnostics(None)
    scipy.optimize.minimize(f, z, method='L-BFGS-B', jac=g,
                            callback=computer.diagnostics,
                            options={'maxiter': maxiter})

if __name__ == '__main__':
    # data_file_names = [os.path.join('graph_US', f'graph{i}.graphml')
    #                    for i in [4, 10, 12, 14, 16, 18, 22]]
    data_file_names = [os.path.join('graph_US', f'graph{i}.graphml')
                       for i in [16]]
    initial_radii = [16.]
    lambda_smooths = [0.0002]
    # lambda_smooths = [0.]
    lambda_geodesics = [0.001, 0.002, 0.004, 0.01, 0.02, 0.04, 0.1, 0.2, 0.4, 1., 2., 4., 10., 20., 40., 100., 200., 400., 1000., 2000., 4000., 10000., 20000., 40000., 100000., 200000., 400000.]
    # lambda_geodesics = [1.]

    arguments = []
    for data_file_name in data_file_names:
        for initial_radius in initial_radii:
            for lambda_smooth in lambda_smooths:
                for lambda_geodesic in lambda_geodesics:
                    arguments.append((
                        data_file_name,
                        1., lambda_smooth, lambda_geodesic,
                        initial_radius, 40, 40, 0.05,
                        1000,
                        os.path.join('..', 'out_leaveout')
                    ))
    with multiprocessing.Pool(50) as p:
        p.starmap(main, arguments)
    # main(*arguments[0])
