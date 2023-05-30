import datetime
import json
import multiprocessing
import os
import pickle
import shutil

import numpy as np
import scipy

from linear_geodesic_optimization import data
from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh
from linear_geodesic_optimization.optimization import optimization

def main(data_file_name, lambda_curvature, lambda_smooth, lambda_geodesic,
         initial_radius, width=20, height=20,
         maxiter=1000, output_dir_name=os.path.join('..', 'out')):
    data_file_path = os.path.join('..', 'data', data_file_name)
    data_name, _ = os.path.splitext(os.path.basename(data_file_name))

    # Construct a mesh
    mesh = RectangleMesh(width, height)

    network_coordinates, network_edges, network_curvatures, network_latencies = data.read_graphml(data_file_path)
    network_vertices = mesh.map_coordinates_to_support(np.array(network_coordinates))
    latencies = data.map_latencies_to_mesh(mesh, network_vertices, network_latencies)

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
            'height': height
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
    data_file_names = [os.path.join('graph_US', f'graph{i}.graphml') for i in [4, 10, 12, 14, 16, 18, 22]]
    initial_radii = [16.]
    lambda_smooths = [0.0002]

    arguments = []
    for data_file_name in data_file_names:
        for initial_radius in initial_radii:
            for lambda_smooth in lambda_smooths:
                arguments.append((
                    data_file_name, 1., lambda_smooth, 0., initial_radius,
                    40, 40,
                    1000,
                    os.path.join('..', 'out_test')
                ))
    # with multiprocessing.Pool(10) as p:
    #     p.starmap(main, arguments)
    main(*arguments[4])