import sys
import time

import numpy as np

sys.path.append('.')
from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh
from linear_geodesic_optimization.optimization.laplacian \
    import Computer as Laplacian
from linear_geodesic_optimization.optimization.curvature \
    import Computer as Curvature


width = 30
height = 30

mesh = RectangleMesh(width, height, extent=1.)
laplacian = Laplacian(mesh)
curvature = Curvature(mesh, laplacian)

rng = np.random.default_rng()
z = mesh.set_parameters(rng.random(width * height))
dz = rng.random(width * height)
dz = 1e-7 * dz / np.linalg.norm(dz)

t = time.time()
curvature.forward()
print(f'Time to compute forward: {time.time() - t}')
kappa_1_z = np.array(curvature.kappa_1)

# Compute the partial derivative in the direction of offset
t = time.time()
curvature.reverse()
print(f'Time to compute reverse: {time.time() - t}')
dif_kappa_1 = np.zeros(kappa_1_z.shape)
for i in range(len(dif_kappa_1)):
    for j, d in curvature.dif_kappa_1[i].items():
        dif_kappa_1[i] += d * dz[j]

# Estimate the partial derivative by adding, evaluating, and subtracting
mesh.set_parameters(z + dz)
curvature.forward()
kappa_1_z_dz = np.array(curvature.kappa_1)
estimated_dif_kappa_1 = kappa_1_z_dz - kappa_1_z

# Print something close to 1., hopefully
quotient = np.linalg.norm(dif_kappa_1) / np.linalg.norm(estimated_dif_kappa_1)
print(f'Quotient of magnitudes: {quotient:.6f}')
# Print something close to 0., hopefully
angle = np.arccos(dif_kappa_1 @ estimated_dif_kappa_1
                  / (np.linalg.norm(dif_kappa_1)
                     * np.linalg.norm(estimated_dif_kappa_1)))
print(f'Angle between:          {angle:.6f}')
