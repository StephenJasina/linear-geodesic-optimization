import datetime
import json
import os

import numpy as np
import scipy

from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh
from linear_geodesic_optimization.optimization import optimization

if __name__ == '__main__':
    toy_directory = os.path.join('..', 'data', 'toy')

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
    directory = os.path.join('..', 'out',
                             datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(directory+'a')
    os.makedirs(directory+'b')

    # Initialize mesh
    fat_edges = mesh.get_fat_edges(network_vertices, network_edges, mesh.get_epsilon())
    positive_sums = np.zeros(width * height)
    positive_counts = np.zeros(width * height)
    for fat_edge, ricci_curvature in zip(fat_edges, ricci_curvatures):
        for vertex in fat_edge:
            positive_sums[vertex] += ricci_curvature
            positive_counts[vertex] += 1
    z = np.zeros(width * height)
    np.divide(positive_sums, positive_counts, out=z, where=(positive_counts != 0))
    z = mesh.set_parameters(z)

    # First optimize without geodesic loss
    hierarchy = optimization.DifferentiationHierarchy(
        mesh, latencies, network_vertices, network_edges, ricci_curvatures,
        lambda_geodesic=0., lambda_curvature=1., lambda_smooth=0.01,
        directory=directory+'a', cores=6)

    f = hierarchy.get_loss_callback()
    g = hierarchy.get_dif_loss_callback()

    hierarchy.diagnostics(None)
    scipy.optimize.minimize(f, z, method='L-BFGS-B', jac=g,
                            callback=hierarchy.diagnostics)

    z = mesh.get_parameters()

    # Then optimize with geodesic loss
    hierarchy = optimization.DifferentiationHierarchy(
        mesh, latencies, network_vertices, network_edges, ricci_curvatures,
        lambda_geodesic=1., lambda_curvature=1., lambda_smooth=0.01,
        directory=directory+'b', cores=6)

    f = hierarchy.get_loss_callback()
    g = hierarchy.get_dif_loss_callback()

    hierarchy.diagnostics(None)
    scipy.optimize.minimize(f, z, method='L-BFGS-B', jac=g,
                            callback=hierarchy.diagnostics)

