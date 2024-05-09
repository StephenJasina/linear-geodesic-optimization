import sys
import time

import dcelmesh
import numpy as np

sys.path.append('.')
from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh
from linear_geodesic_optimization.optimization.curvature \
    import Computer as Curvature
from linear_geodesic_optimization.optimization.laplacian \
    import Computer as Laplacian
from linear_geodesic_optimization.optimization.smooth_loss \
    import Computer as SmoothLoss


width = 50
height = 50

mesh = RectangleMesh(width, height)
laplacian = Laplacian(mesh)
curvature = Curvature(mesh, laplacian)
smooth = SmoothLoss(mesh, laplacian, curvature)

seed = time.time_ns()
seed = seed % (2**32 - 1)
np.random.seed(seed)
print(f'Seed: {seed}')
z = mesh.set_parameters(np.random.random(width * height))
# z = mesh.set_parameters(np.array([
#     (20.**2
#      - mesh.get_coordinates()[index][0]**2
#      - mesh.get_coordinates()[index][1]**2
#     )**0.5
#     for index in range(mesh.get_topology().n_vertices())
# ]))
dz = np.random.random(width * height)
dz = dz / np.linalg.norm(dz)
h = 1e-9

t = time.time()
smooth.forward()
print(f'Time to compute forward: {time.time() - t}')
loss_z = smooth.loss

# Compute the partial derivative in the direction of offset
t = time.time()
smooth.reverse()
print(f'Time to compute reverse: {time.time() - t}')
dif_loss = smooth.dif_loss @ dz

# Estimate the partial derivative by adding, evaluating, and subtracting
mesh.set_parameters(z + h * dz)
smooth.forward()
loss_z_plus_dz = smooth.loss
mesh.set_parameters(z - h * dz)
smooth.forward()
loss_z_minus_dz = smooth.loss
estimated_dif_loss = (loss_z_plus_dz - loss_z_minus_dz) / (2. * h)

# Should print something close to 1
print(f'Quotient: {dif_loss / estimated_dif_loss:.6f}')
