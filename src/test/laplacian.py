import time

import numpy as np

import linear_geodesic_optimization.mesh.rectangle
from linear_geodesic_optimization.optimization.laplacian \
    import Computer as Laplacian


width = 40
height = 40
np.random.seed(0)

mesh = linear_geodesic_optimization.mesh.rectangle.Mesh(width, height)
z = mesh.set_parameters(np.random.random(width * height))
dz = np.random.random(width * height)
dz = 1e-7 * dz / np.linalg.norm(dz)

laplacian = Laplacian(mesh)

laplacian.forward()
LC_dirichlet_vertices_z = np.array(laplacian.LC_dirichlet_vertices)

# Compute the partial derivative in the direction of offset
laplacian.reverse()
dif_LC_dirichlet_vertices = np.zeros(LC_dirichlet_vertices_z.shape)
for i in range(len(dif_LC_dirichlet_vertices)):
    for j, d in laplacian.dif_LC_dirichlet_vertices[i].items():
        dif_LC_dirichlet_vertices[i] += d * dz[j]

# Estimate the partial derivative by adding, evaluating, and subtracting
mesh.set_parameters(z + dz)
laplacian.forward()
LC_dirichlet_vertices_z_dz = np.array(laplacian.LC_dirichlet_vertices)
estimated_dif_LC_dirichlet_vertices = LC_dirichlet_vertices_z_dz - LC_dirichlet_vertices_z

for true, estimated in zip(dif_LC_dirichlet_vertices, estimated_dif_LC_dirichlet_vertices):
    print(true, estimated)
