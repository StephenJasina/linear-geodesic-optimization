import sys
import time

import numpy as np

sys.path.append('.')
from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh
from linear_geodesic_optimization.optimization.laplacian \
    import Computer as Laplacian
from linear_geodesic_optimization.optimization.curvature \
    import Computer as Curvature


width = 10
height = 10

mesh = RectangleMesh(width, height)
laplacian = Laplacian(mesh)
curvature = Curvature(mesh, laplacian)

# z = mesh.set_parameters(np.random.random(width * height))
z = mesh.set_parameters(np.array([
    (16.**2
     - mesh.get_coordinates()[index][0]**2
     - mesh.get_coordinates()[index][1]**2
    )**0.5
    for index in range(mesh.get_topology().n_vertices())
]))
dz = np.random.random(width * height)
dz = 1e-7 * dz / np.linalg.norm(dz)

t = time.time()
curvature.forward()
print(f'Time to compute forward: {time.time() - t}')
kappa_H_z = np.array(curvature.kappa_H)

# # Compute the partial derivative in the direction of offset
# t = time.time()
# curvature.reverse()
# print(f'Time to compute reverse: {time.time() - t}')
# dif_kappa_H = np.zeros(kappa_H_z.shape)
# for i in range(len(dif_kappa_H)):
#     for j, d in curvature.dif_kappa_H[i].items():
#         dif_kappa_H[i] += d * dz[j]

# # Estimate the partial derivative by adding, evaluating, and subtracting
# mesh.set_parameters(z + dz)
# curvature.forward()
# kappa_H_z_dz = np.array(curvature.kappa_H)
# estimated_dif_kappa_H = kappa_H_z_dz - kappa_H_z

# # for true, estimated in zip(dif_kappa_H, estimated_dif_kappa_H):
# #     print(true, estimated)

# # Print something close to 1., hopefully
# worst_deviation = np.exp(np.amax(np.abs(np.log(dif_kappa_H / estimated_dif_kappa_H))))
# print(f'Greatest deviation: {worst_deviation}')
