import datetime
import json
import multiprocessing
import os
import pickle

import numpy as np
import scipy

from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh
from linear_geodesic_optimization.optimization import optimization

def main(data_name, lambda_geodesic, lambda_curvature, lambda_smooth, initial_radius,
         smoothness_strategy='mean', width=20, height=20, maxiter=1000):
    data_directory = os.path.join('..', 'data', data_name)

    # Construct a mesh
    mesh = RectangleMesh(width, height)
    vertices = mesh.get_vertices()
    V = vertices.shape[0]

    coordinates = None
    label_to_index = {}
    with open(os.path.join(data_directory, 'position.json')) as f:
        position_json = json.load(f)

        label_to_index = {label: index for index, label in enumerate(position_json)}

        coordinates = [None for _ in range(len(position_json))]
        for vertex, position in position_json.items():
            coordinates[label_to_index[vertex]] = position

    network_vertices = mesh.map_coordinates_to_support(coordinates)

    network_edges = []
    latencies = {mesh.nearest_vertex_index(network_vertices[i]): [] for i in range(len(network_vertices))}
    with open(os.path.join(data_directory, 'latency.json')) as f:
        latency_json = json.load(f)

        for edge, latency in latency_json.items():
            u = label_to_index[edge[0]]
            v = label_to_index[edge[1]]

            network_edges.append((u, v))
            latencies[mesh.nearest_vertex_index(network_vertices[u])].append(
                (mesh.nearest_vertex_index(network_vertices[v]), latency)
            )

    network_curvatures = []
    with open(os.path.join(data_directory, 'curvature.json')) as f:
        network_curvatures = list(json.load(f).values())

    # Setup snapshots
    directory = os.path.join(
        '..', 'out', f'{data_name}', f'{smoothness_strategy}',
        f'{lambda_geodesic}_{lambda_curvature}_{lambda_smooth}_{initial_radius}_{width}_{height}'
    )
    os.makedirs(directory)

    with open(os.path.join(directory, 'parameters'), 'wb') as f:
        pickle.dump({
            'data_name': data_name,
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
        for i in range(width)
        for j in range(height)
    ]).reshape((width * height,))
    z = z - np.amin(z)
    z = mesh.set_parameters(z)

    hierarchy = optimization.DifferentiationHierarchy(
        mesh, latencies, network_vertices, network_edges, network_curvatures,
        lambda_geodesic, lambda_curvature, lambda_smooth, smoothness_strategy,
        directory=directory, cores=None)

    f = hierarchy.get_loss_callback()
    g = hierarchy.get_dif_loss_callback()

    hierarchy.diagnostics(None)
    scipy.optimize.minimize(f, z, method='L-BFGS-B', jac=g,
                            callback=hierarchy.diagnostics,
                            options={'maxiter': maxiter})

if __name__ == '__main__':
    arguments = []
    for smoothness_strategy in ['gaussian', 'mean', 'mvs_cross']:
        for initial_radius in [1., 2., 4., 8., 16.]:
            for lambda_smooth in [0., 0.001, 0.002, 0.004, 0.01, 0.02]:
                arguments.append(('elbow', 0., 1., lambda_smooth, initial_radius, smoothness_strategy))
    with multiprocessing.Pool(45) as p:
        p.starmap(main, arguments)
