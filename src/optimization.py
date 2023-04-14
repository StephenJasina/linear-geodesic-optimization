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

def main(data_file_name, lambda_geodesic, lambda_curvature, lambda_smooth, initial_radius,
         smoothness_strategy='mean', width=20, height=20,
         maxiter=1000, output_dir_name=os.path.join('..', 'out')):
    data_file_path = os.path.join('..', 'data', data_file_name)
    data_name, data_type = os.path.splitext(os.path.basename(data_file_name))

    # Construct a mesh
    mesh = RectangleMesh(width, height)
    vertices = mesh.get_vertices()
    V = vertices.shape[0]

    if data_type == '.json':
        coordinates, network_edges, network_curvatures, network_latencies = data.read_json(data_file_path)
    elif data_type == '.graphml':
        coordinates, network_edges, network_curvatures, network_latencies = data.read_graphml(data_file_path)
    network_vertices = mesh.map_coordinates_to_support(coordinates)
    latencies = data.map_latencies_to_mesh(mesh, network_vertices, network_latencies)

    # Setup snapshots
    directory = os.path.join(
        output_dir_name, f'{data_name}', f'{smoothness_strategy}',
        f'{lambda_geodesic}_{lambda_curvature}_{lambda_smooth}_{initial_radius}_{width}_{height}'
    )
    if os.path.isdir(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

    with open(os.path.join(directory, 'parameters'), 'wb') as f:
        pickle.dump({
            'data_file_name': data_file_name,
            'lambda_geodesic': lambda_geodesic,
            'lambda_curvature': lambda_curvature,
            'lambda_smooth': lambda_smooth,
            'initial_radius': initial_radius,
            'smoothness_strategy': smoothness_strategy,
            'width': width,
            'height': height
        }, f)

    # Initialize mesh
    z = np.array([
        (initial_radius**2
            - (i / (width - 1) - 0.5)**2
            - (j / (height - 1) - 0.5)**2)**0.5
        for j in range(height)
        for i in range(width)
    ]).reshape((width * height,))
    z = z - np.amin(z)
    z = mesh.set_parameters(z)

    hierarchy = optimization.DifferentiationHierarchy(
        mesh, latencies, network_vertices, network_edges, network_curvatures,
        lambda_geodesic, lambda_curvature, lambda_smooth, smoothness_strategy,
        directory=directory)

    f = hierarchy.get_loss_callback()
    g = hierarchy.get_dif_loss_callback()

    hierarchy.diagnostics(None)
    scipy.optimize.minimize(f, z, method='L-BFGS-B', jac=g,
                            callback=hierarchy.diagnostics,
                            options={'maxiter': maxiter})

if __name__ == '__main__':
    data_file_names = [os.path.join('graph_US', f'graph{i}.graphml') for i in [4, 10, 12, 14, 16, 18, 22]]
    initial_radii = [16.]
    lambda_smooths = [0.0002]

    arguments = []
    for data_file_name in data_file_names:
        for smoothness_strategy in ['mvs_cross']:
            for initial_radius in initial_radii:
                for lambda_smooth in lambda_smooths:
                    arguments.append((
                        data_file_name, 0., 1., lambda_smooth, initial_radius,
                        smoothness_strategy,
                        40, 40,
                        1000,
                        os.path.join('..', 'out_US')
                    ))
    with multiprocessing.Pool(10) as p:
        p.starmap(main, arguments)
