import sys
import time

import numpy as np

sys.path.append('.')
from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh
from linear_geodesic_optimization.optimization.geodesic import Computer as Geodesic
from linear_geodesic_optimization.optimization.laplacian import Computer as Laplacian

seed = time.time_ns()
seed = seed % (2**32 - 1)
print(f'Seed: {seed}')
rng = np.random.default_rng(seed)

width = 20
height = 20

mesh = RectangleMesh(width, height)
topology = mesh.get_topology()
laplacian = Laplacian(mesh)
source_index = mesh.nearest_vertex(np.array([0., 0.])).index
target_indices = [
    mesh.nearest_vertex(np.array([0.25, 0.25])).index,
    mesh.nearest_vertex(np.array([0.25, -0.25])).index,
]
geodesic = Geodesic(mesh, source_index, target_indices, laplacian, 1000.)

z = mesh.set_parameters(rng.random(width * height))
dz = rng.random(width * height)
dz = 1e-7 * dz / np.linalg.norm(dz)

def f(geodesic: Geodesic, z):
    mesh.set_parameters(z)
    geodesic.forward()
    return geodesic.distances[target_indices[0]]

def g(geodesic: Geodesic, z, dz):
    mesh.set_parameters(z)
    geodesic.reverse()
    return geodesic.dif_distances[target_indices[0]] @ dz / np.linalg.norm(dz)

t = time.time()
quantity_z = np.array(f(geodesic, z)).flatten()
print(f'Time to compute forward: {time.time() - t}')

t = time.time()
dif_quantity = np.array(g(geodesic, z, dz)).flatten()
print(f'Time to compute reverse: {time.time() - t}')

quantity_z_dz = np.array(f(geodesic, z + dz)).flatten()
estimated_dif_quantity = (quantity_z_dz - quantity_z) / np.linalg.norm(dz)

quotient = np.linalg.norm(dif_quantity) / np.linalg.norm(estimated_dif_quantity)
print(f'Quotient of magnitudes: {quotient:.6f}')

angle = np.arccos(dif_quantity @ estimated_dif_quantity
                  / (np.linalg.norm(dif_quantity)
                     * np.linalg.norm(estimated_dif_quantity)))
print(f'Angle between:          {angle:.6f}')
