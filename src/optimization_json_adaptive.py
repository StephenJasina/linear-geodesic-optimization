import datetime
import json
import multiprocessing
import os
import pickle
import shutil

import numpy as np
import scipy

from linear_geodesic_optimization.mesh.adaptive import Mesh as AdaptiveMesh
from linear_geodesic_optimization.optimization import optimization

def main(data_name, lambda_geodesic, lambda_curvature, lambda_smooth, initial_radius,
         smoothness_strategy='mean', width=8, height=8, fat_edges_only=False,
         maxiter=1000, output_dir_name=os.path.join('..', 'out')):
    data_directory = os.path.join('..', 'data', data_name)

    # Get the network vertices
    coordinates = None
    label_to_index = {}
    with open(os.path.join(data_directory, 'position.json')) as f:
        position_json = json.load(f)

        label_to_index = {label: index for index, label in enumerate(position_json)}

        coordinates = [None for _ in range(len(position_json))]
        for vertex, position in position_json.items():
            coordinates[label_to_index[vertex]] = position

    network_vertices = AdaptiveMesh.map_coordinates_to_support(coordinates)

    # Get the network edges and curvatures
    network_edges = []
    network_curvatures = []
    with open(os.path.join(data_directory, 'curvature.json')) as f:
        curvature_json = json.load(f)

        for edge, network_curvature in curvature_json.items():
            u = label_to_index[edge[0]]
            v = label_to_index[edge[1]]

            network_edges.append((u, v))
            network_curvatures.append(network_curvature)

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

    latencies = {mesh.nearest_vertex_index(network_vertices[i]): [] for i in range(len(network_vertices))}
    with open(os.path.join(data_directory, 'latency.json')) as f:
        latency_json = json.load(f)

        for edge, latency in latency_json.items():
            u = label_to_index[edge[0]]
            v = label_to_index[edge[1]]

            latencies[mesh.nearest_vertex_index(network_vertices[u])].append(
                (mesh.nearest_vertex_index(network_vertices[v]), latency)
            )

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
            'data_name': data_name,
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
                    'elbow', 0., 1., lambda_smooth, initial_radius,
                    smoothness_strategy,
                    20, 20, True,
                    1000,
                    os.path.join('..', 'out_fat_edges_only')
                ))
    with multiprocessing.Pool(1) as p:
        p.starmap(main, arguments)
