import sys
import time

import numpy as np

sys.path.append('.')
from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh
from linear_geodesic_optimization.optimization.laplacian \
    import Computer as Laplacian


width = 20
height = 20

mesh = RectangleMesh(width, height, extent=1.)
laplacian = Laplacian(mesh)

z = mesh.set_parameters(np.random.random(width * height))
dz = np.random.random(width * height)
dz = 1e-7 * dz / np.linalg.norm(dz)

t = time.time()
laplacian.forward()
print(f'Time to compute forward: {time.time() - t}')
LC_neumann_vertices_z = np.array(laplacian.LC_neumann_vertices)

# Compute the partial derivative in the direction of offset
t = time.time()
laplacian.reverse()
print(f'Time to compute reverse: {time.time() - t}')
dif_LC_neumann_vertices = np.zeros(LC_neumann_vertices_z.shape)
for i in range(len(dif_LC_neumann_vertices)):
    for j, d in laplacian.dif_LC_neumann_vertices[i].items():
        dif_LC_neumann_vertices[i] += d * dz[j]

# Estimate the partial derivative by adding, evaluating, and subtracting
mesh.set_parameters(z + dz)
laplacian.forward()
LC_neumann_vertices_z_dz = np.array(laplacian.LC_neumann_vertices)
estimated_dif_LC_neumann_vertices = LC_neumann_vertices_z_dz - LC_neumann_vertices_z

# for true, estimated in zip(dif_LC_neumann_vertices, estimated_dif_LC_neumann_vertices):
#     print(true, estimated)

# Print something close to 1., hopefully
quotient = np.linalg.norm(dif_LC_neumann_vertices) / np.linalg.norm(estimated_dif_LC_neumann_vertices)
print(f'Quotient of magnitudes: {quotient:.6f}')
# Print something close to 0., hopefully
angle = np.arccos(dif_LC_neumann_vertices @ estimated_dif_LC_neumann_vertices
                  / (np.linalg.norm(dif_LC_neumann_vertices)
                     * np.linalg.norm(estimated_dif_LC_neumann_vertices)))
print(f'Angle between:          {angle:.6f}')
