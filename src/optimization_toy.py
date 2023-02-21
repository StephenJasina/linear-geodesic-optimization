import datetime
import json
import multiprocessing
import os

import numpy as np
import scipy

from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh
from linear_geodesic_optimization.optimization import optimization

def main(lambda_geodesic, lambda_curvature, lambda_smooth, initial_radius):
    toy_directory = os.path.join('..', 'data', 'two_islands_mean')

    # Construct a mesh
    width = 20
    height = 20
    mesh = RectangleMesh(width, height)
    vertices = mesh.get_vertices()
    V = vertices.shape[0]

    coordinates = None
    label_to_index = {}
    with open(os.path.join(toy_directory, 'position.json')) as f:
        position_json = json.load(f)

        label_to_index = {label: index for index, label in enumerate(position_json)}

        coordinates = [None for _ in range(len(position_json))]
        for vertex, position in position_json.items():
            coordinates[label_to_index[vertex]] = position

    network_vertices = mesh.map_coordinates_to_support(coordinates)

    network_edges = []
    latencies = {mesh.nearest_vertex_index(network_vertices[i]): [] for i in range(len(network_vertices))}
    with open(os.path.join(toy_directory, 'latency.json')) as f:
        latency_json = json.load(f)

        for edge, latency in latency_json.items():
            u = label_to_index[edge[0]]
            v = label_to_index[edge[1]]

            network_edges.append((u, v))
            latencies[mesh.nearest_vertex_index(network_vertices[u])].append(
                (mesh.nearest_vertex_index(network_vertices[v]), latency)
            )

    ricci_curvatures = []
    with open(os.path.join(toy_directory, 'ricci_curvature.json')) as f:
        ricci_curvatures = list(json.load(f).values())

    # Setup snapshots
    # directory = os.path.join('..', 'out',
    #                          datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    directory = os.path.join(
        '..', 'out', 'two_islands',
        f'{lambda_geodesic}_{lambda_curvature}_{lambda_smooth}_{initial_radius}_{width}_{height}'
    )
    os.makedirs(directory)

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
        mesh, latencies, network_vertices, network_edges, ricci_curvatures,
        lambda_geodesic, lambda_curvature, lambda_smooth, 'mean',
        directory=directory, cores=None)

    f = hierarchy.get_loss_callback()
    g = hierarchy.get_dif_loss_callback()

    hierarchy.diagnostics(None)
    scipy.optimize.minimize(f, z, method='L-BFGS-B', jac=g,
                            callback=hierarchy.diagnostics,
                            options={'maxiter': 100})

if __name__ == '__main__':
    arguments = []
    for initial_radius in [1., 2., 4., 8., 16.]:
        for lambda_smooth in [0.001, 0.002, 0.004, 0.01, 0.02]:
            arguments.append((0., 1., lambda_smooth, initial_radius))
    with multiprocessing.Pool(25) as p:
        p.starmap(main, arguments)
