import datetime
import os

import networkx as nx
import numpy as np
import scipy.optimize

from linear_geodesic_optimization.data import phony
from linear_geodesic_optimization.mesh.sphere import Mesh as SphereMesh
from linear_geodesic_optimization.optimization import optimization

if __name__ == '__main__':
    # Get graph data first

    # Good global graphs are DeutcheTelekom, Highwinds, HostwayInternational,
    # HurricaneElectric, Ntt, and Peer1
    network = nx.read_graphml(os.path.join('..', 'data',
                                           'HostwayInternational.graphml'))

    latitudes = [network.nodes[v]['Latitude'] for v in network.nodes]
    longitudes = [network.nodes[v]['Longitude'] for v in network.nodes]
    network_vertices = [SphereMesh.latitude_longitude_to_direction(latitude,
                                                                   longitude)
                        for latitude, longitude in zip(latitudes, longitudes)]
    network_edges = [(int(u), int(v)) for u, v in network.edges]
    ricci_curvatures = [network.edges[e]['ricciCurvature']
                        for e in network.edges]

    # Compute phony latencies
    c = 299792458
    r = 6367500
    for u, v in network.edges:
        network.edges[(u, v)]['latency'] = \
            2000 * r * np.arccos(network_vertices[int(u)] @ network_vertices[int(v)]) / (3 * c)

    latencies = np.zeros((len(network.nodes), len(network.nodes)))
    for u, u_paths in nx.shortest_path(network).items():
        for v, path in u_paths.items():
            latencies[int(u), int(v)] = sum(network.edges[(path[i], path[i + 1])]['latency']
                                            for i in range(len(path) - 1))

    # Setup snapshots
    directory = os.path.join('..', 'out',
                             datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(directory)

    # Construct the mesh
    frequency = 2
    mesh = SphereMesh(frequency)
    partials = mesh.get_partials()
    V = partials.shape[0]
    rho = mesh.get_parameters()

    # Get some (phony) latency measurements
    ts = phony.sphere_random(mesh)

    hierarchy = optimization.DifferentiationHierarchy(
        mesh, ts, network_vertices, network_edges, ricci_curvatures,
        lambda_geodesic=1., lambda_curvature=1., lambda_smooth=0.01,
        directory=directory)

    f = hierarchy.get_loss_callback()
    g = hierarchy.get_dif_loss_callback()

    hierarchy.diagnostics(None)
    scipy.optimize.minimize(f, rho, method='L-BFGS-B', jac=g,
                            callback=hierarchy.diagnostics)
