import sys

import dcelmesh
import numpy as np

sys.path.append('.')
from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh
from linear_geodesic_optimization.optimization.geodesic \
    import Computer as Geodesic
from linear_geodesic_optimization.optimization.laplacian \
    import Computer as Laplacian


width = 4
height = 4

mesh = RectangleMesh(width, height, extent=1.)
geodesic = Geodesic(mesh, 3, 12)

np.random.seed(0)
z = mesh.set_parameters(np.random.random(width * height))
# z = mesh.set_parameters(np.ones(width * height))
dz = np.random.random(width * height)
dz = np.zeros(width * height)
dz[2] = 1.
dz = dz / np.linalg.norm(dz)
h = 1e-7

geodesic.forward()
distance_z = geodesic.distance
print(f'Total distance: {distance_z}')
print('Path:')
print(f'\t{geodesic.path_vertices[0].index()}')
for vertex, halfedges in zip(geodesic.path_vertices[1:],
                             geodesic.path_halfedges):
    for halfedge in halfedges:
        print(f'\t({halfedge.origin().index()}, {halfedge.destination().index()})')
    print(f'\t{vertex.index()}')

# Compute the partial derivative in the direction of offset
geodesic.reverse()
dif_distance = np.float64(0.)
for j, d in geodesic.dif_distance.items():
    dif_distance += d * dz[j]

# Estimate the partial derivative by adding, evaluating, and subtracting
mesh.set_parameters(z + h * dz)
geodesic.forward()
distance_z_plus_dz = geodesic.distance
mesh.set_parameters(z - h * dz)
geodesic.forward()
distance_z_minus_dz = geodesic.distance
estimated_dif_distance = (distance_z_plus_dz - distance_z_minus_dz) / (2. * h)

# Should print something close to 1
print(f'Quotient: {dif_distance / estimated_dif_distance:.6f}')
print(f'{dif_distance}')
print(f'{estimated_dif_distance}')