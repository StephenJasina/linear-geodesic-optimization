import sys

import dcelmesh
import numpy as np

sys.path.append('.')
from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh
from linear_geodesic_optimization.optimization.geodesic \
    import Computer as Geodesic
from linear_geodesic_optimization.optimization.laplacian \
    import Computer as Laplacian


width = 30
height = 30

mesh = RectangleMesh(width, height)
geodesic = Geodesic(mesh, 0, 123)

np.random.seed(0)
z = mesh.set_parameters(np.random.random(width * height))
# dz = np.random.random(width * height)
dz = np.zeros(width * height)
dz[0] = 1.
dz = dz / np.linalg.norm(dz)
h = 1e-7

geodesic.forward()
distance_z = geodesic.distance
print(f'Total distance: {distance_z}')
print('Path:')
for element in geodesic.path:
    if isinstance(element, dcelmesh.Mesh.Vertex):
        print(f'\t{element.index()}')
    else:
        print(f'\t({element.origin().index()}, {element.destination().index()})')

# Compute the partial derivative in the direction of offset
geodesic.reverse()
dif_distance = np.float64(0.)
for j, d in geodesic.dif_distance.items():
    dif_distance += d * dz[j]

# Estimate the partial derivative by adding, evaluating, and subtracting
mesh.set_parameters(z + h * dz)
geodesic.forward()
distance_z_dz = geodesic.distance
estimated_dif_distance = (distance_z_dz - distance_z) / h

print(dif_distance, estimated_dif_distance)

# geodesic.forward()
# edge_lengths_z = np.array(geodesic.edge_lengths)

# # Compute the partial derivative in the direction of offset
# geodesic.reverse()
# dif_edge_lengths = np.zeros(edge_lengths_z.shape)
# for i in range(len(dif_edge_lengths)):
#     for j, d in geodesic.dif_edge_lengths[i].items():
#         dif_edge_lengths[i] += d * dz[j]

# # Estimate the partial derivative by adding, evaluating, and subtracting
# mesh.set_parameters(z + h * dz)
# geodesic.forward()
# edge_lengths_z_dz = np.array(geodesic.edge_lengths)
# estimated_dif_edge_lengths = (edge_lengths_z_dz - edge_lengths_z) / h

# for true, estimated in zip(dif_edge_lengths, estimated_dif_edge_lengths):
#     print(true, estimated)
