import datetime
import json
import os

import numpy as np
import scipy

from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh
from linear_geodesic_optimization.optimization import optimization

if __name__ == '__main__':
    # Construct a mesh
    width = 5
    height = 5
    mesh = RectangleMesh(width, height)
    vertices = mesh.get_vertices()
    V = vertices.shape[0]
    z = mesh.get_parameters()

    # TODO: Do something smarter here
    # Make the mesh a little noisy for symmetry breaking
    rng = np.random.default_rng(0)
    z = mesh.set_parameters(rng.random(width * height) / 100)

    coordinates = None
    label_to_index = {}
    with open(os.path.join('..', 'data', 'toy', 'position.json')) as f:
        position_json = json.load(f)

        label_to_index = {label: index for index, label in enumerate(position_json)}

        coordinates = [None for _ in range(len(position_json))]
        for vertex, position in position_json.items():
            coordinates[label_to_index[vertex]] = position

    network_vertices = mesh.scale_coordinates_to_unit_square(coordinates)

    network_edges = []
    ts = {i: [] for i in range(len(network_vertices))}
    with open(os.path.join('..', 'data', 'toy', 'latency.json')) as f:
        latency_json = json.load(f)

        for edge, latency in latency_json.items():
            u = label_to_index[edge[0]]
            v = label_to_index[edge[1]]

            network_edges.append((u, v))

            ts[u].append((v, latency))
            ts[v].append((u, latency))

    ricci_curvatures = []
    with open(os.path.join('..', 'data', 'toy', 'ricci_curvature.json')) as f:
        ricci_curvatures = list(json.load(f).values())

    # Setup snapshots
    directory = os.path.join('..', 'out',
                             datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(directory)

    hierarchy = optimization.DifferentiationHierarchy(
        mesh, ts, network_vertices, network_edges, ricci_curvatures,
        lambda_geodesic=0., lambda_curvature=1., lambda_smooth=0.01,
        directory=directory)

    f = hierarchy.get_loss_callback()
    g = hierarchy.get_dif_loss_callback()

    hierarchy.diagnostics(None)
    scipy.optimize.minimize(f, z, method='L-BFGS-B', jac=g,
                            callback=hierarchy.diagnostics,
                            options=dict(maxiter=100))
