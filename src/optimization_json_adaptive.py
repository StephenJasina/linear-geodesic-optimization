import datetime
import json
import multiprocessing
import os
import pickle
import shutil

import numpy as np
import scipy

from linear_geodesic_optimization import data
from linear_geodesic_optimization.mesh.adaptive import Mesh as AdaptiveMesh
from linear_geodesic_optimization.optimization import optimization

def main(data_file_name, lambda_geodesic, lambda_curvature, lambda_smooth, initial_radius,
         smoothness_strategy='mean', width=8, height=8, fat_edges_only=False,
         maxiter=1000, output_dir_name=os.path.join('..', 'out')):
    data_file_path = os.path.join('..', 'data', data_file_name)
    data_name, data_type = os.path.splitext(os.path.basename(data_file_name))

    coordinates, network_edges, network_curvatures, network_latencies = data.read_json(data_file_path)
    network_vertices = AdaptiveMesh.map_coordinates_to_support(coordinates)

    # Create the mesh
    density = np.amin([
        np.linalg.norm(network_vertices[i] - network_vertices[j])
        for i, j in network_edges
    ]) / 10
    points = list(np.array(network_vertices))
    for i, j in network_edges:
        count = int(np.ceil(np.linalg.norm(network_vertices[i] - network_vertices[j]) / density))
        for p in np.linspace(network_vertices[i], network_vertices[j], count)[1:-1]:
            # Guard for floating point error
            if np.amin([np.linalg.norm(p - point) for point in points]) > 1e-10:
                points.append(p)
    mesh = AdaptiveMesh(width, height, points)

    if fat_edges_only:
        fat_edges = mesh.get_fat_edges(network_vertices, network_edges, mesh.get_epsilon() / 2.)
        mesh.restrict_to_fat_edges(fat_edges)

    latencies = data.map_latencies_to_mesh(mesh, network_vertices, network_latencies)

    # Setup snapshots
    directory = os.path.join(
        output_dir_name, f'{data_name}', f'{smoothness_strategy}',
        f'{lambda_geodesic}_{lambda_curvature}_{lambda_smooth}_{initial_radius}'
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
            'height': height,
            'fat_edges_only': fat_edges_only
        }, f)

    # Initialize mesh
    z = np.array([
        (initial_radius**2 - x**2 - y**2)**0.5
        for x, y, _ in mesh.get_vertices()
    ])
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
    arguments = []
    for smoothness_strategy in ['mean']:
        for initial_radius in [16.]:
            for lambda_smooth in [0.001]:
                arguments.append((
                    'elbow.json', 0., 1., lambda_smooth, initial_radius,
                    smoothness_strategy,
                    20, 20, True,
                    1000,
                    os.path.join('..', 'out_test')
                ))
    with multiprocessing.Pool(1) as p:
        p.starmap(main, arguments)
