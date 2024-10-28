import os
import sys
import time

import networkx as nx
import numpy as np

sys.path.append('.')
from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh
from linear_geodesic_optimization.optimization.laplacian \
    import Computer as Laplacian
from linear_geodesic_optimization.optimization.curvature \
    import Computer as Curvature
from linear_geodesic_optimization.optimization.curvature_loss \
    import Computer as CurvatureLoss
from linear_geodesic_optimization.data import input_network


seed = time.time_ns()
seed = seed % (2**32 - 1)
np.random.seed(seed)

width = 20
height = 20
mesh = RectangleMesh(width, height, extent=1.)

graph_data, vertex_data, edge_data = input_network.get_network_data(
    nx.read_graphml(
        os.path.join('..', 'data', 'ipv4', 'graph_US', 'graph16.graphml')
    )
)
coordinates = graph_data['coordinates']
network_edges = graph_data['edges']
network_curvatures = edge_data['ricciCurvature']
network_vertices = mesh.map_coordinates_to_support(
    np.array(coordinates, dtype=np.float64)
)

laplacian = Laplacian(mesh)
curvature = Curvature(mesh, laplacian)
curvature_loss = CurvatureLoss(
    mesh, network_vertices, network_edges,
    network_curvatures, 1.01 * 2**0.5 / width,
    curvature, np.random.random(len(network_edges))
)

z = mesh.set_parameters(np.random.random(width * height) / 100)
# h is smaller here than in other tests since the values are more
# resilient to small changes
h = 1e-4
dz = np.random.random(width * height)
dz = h * dz / np.linalg.norm(dz)

t = time.time()
curvature_loss.forward()
print(f'Time to compute forward: {time.time() - t}')
loss_z = curvature_loss.loss

# Compute the partial derivative in the direction of offset
t = time.time()
curvature_loss.reverse()
print(f'Time to compute reverse: {time.time() - t}')
dif_loss = np.float64(0.)
for j, d in enumerate(curvature_loss.dif_loss):
    dif_loss += d * dz[j]

# Estimate the partial derivative by adding, evaluating, and subtracting
mesh.set_parameters(z + h * dz)
curvature_loss.forward()
loss_z_plus_dz = curvature_loss.loss
mesh.set_parameters(z - h * dz)
curvature_loss.forward()
loss_z_minus_dz = curvature_loss.loss
estimated_dif_loss = (loss_z_plus_dz - loss_z_minus_dz) / (2. * h)

# Should print something close to 1
print(f'Quotient: {dif_loss / estimated_dif_loss:.6f}')
