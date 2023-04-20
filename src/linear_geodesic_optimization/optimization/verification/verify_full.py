import datetime
import json
import os

import numpy as np
import scipy

from linear_geodesic_optimization import data
from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh
from linear_geodesic_optimization.optimization import optimization

if __name__ == '__main__':
    toy_file_path = os.path.join('..', 'data', 'toy.json')

    smooth_strategy = 'mvs-cross'

    # Construct a mesh
    width = 8
    height = 8
    mesh = RectangleMesh(width, height)
    vertices = mesh.get_vertices()
    V = vertices.shape[0]

    coordinates, network_edges, network_curvatures, network_latencies = data.read_json(toy_file_path)
    network_vertices = mesh.map_coordinates_to_support(coordinates)
    latencies = data.map_latencies_to_mesh(mesh, network_vertices, network_latencies)

    rng = np.random.default_rng(0)
    z_0 = rng.random(V)

    hierarchy = optimization.DifferentiationHierarchy(
        mesh, latencies, network_vertices, network_edges, network_curvatures,
        lambda_geodesic=1., lambda_curvature=1., lambda_smooth=1.,
        smooth_strategy=smooth_strategy)

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
