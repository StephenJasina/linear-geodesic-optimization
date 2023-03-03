import datetime
import json
import os

import numpy as np
import scipy

from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh
from linear_geodesic_optimization.optimization import optimization

if __name__ == '__main__':
    toy_directory = os.path.join('..', 'data', 'toy')

    smooth_strategy = 'gaussian'

    # Construct a mesh
    width = 4
    height = 4
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
    ts = {i: [] for i in range(len(network_vertices))}
    with open(os.path.join(toy_directory, 'latency.json')) as f:
        latency_json = json.load(f)

        for edge, latency in latency_json.items():
            u = label_to_index[edge[0]]
            v = label_to_index[edge[1]]

            network_edges.append((u, v))

            ts[u].append((v, latency))

    network_curvatures = []
    with open(os.path.join(toy_directory, 'curvature.json')) as f:
        network_curvatures = list(json.load(f).values())

    rng = np.random.default_rng(0)
    z_0 = rng.random(V)

    hierarchy = optimization.DifferentiationHierarchy(
        mesh, ts, network_vertices, network_edges, network_curvatures,
        lambda_geodesic=1., lambda_curvature=1., lambda_smooth=1.,
        smooth_strategy=smooth_strategy, cores=None)

    f = hierarchy.get_loss_callback()
    g = hierarchy.get_dif_loss_callback()

    delta = 1e-5

    f_0 = f(z_0)
    g_0 = g(z_0)

    for l in range(V):
        z_delta = np.copy(z_0)
        z_delta[l] += delta
        f_delta = f(z_delta)

        print((f_delta - f_0) / delta, g_0[l])
