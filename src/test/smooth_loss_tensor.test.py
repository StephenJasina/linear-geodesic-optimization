import sys
import time

import numpy as np

sys.path.append('.')
from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh
from linear_geodesic_optimization.optimization.laplacian \
    import Computer as Laplacian
from linear_geodesic_optimization.optimization.smooth_loss_tensor \
    import Computer as SmoothLoss


width = 30
height = 30

mesh = RectangleMesh(width, height)
laplacian = Laplacian(mesh)
smooth = SmoothLoss(mesh, laplacian)

seed = time.time_ns()
seed = seed % (2**32 - 1)
np.random.seed(seed)
print(f'Seed: {seed}')
# Steep enough that the slope-dependent terms matter
z = mesh.set_parameters(0.1 * np.random.random(width * height))
dz = np.random.random(width * height)
dz = dz / np.linalg.norm(dz)
h = 1e-7

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

print(f'Loss: {loss_z}')
print(f'Analytic directional derivative: {dif_loss}')
print(f'Estimated directional derivative: {estimated_dif_loss}')
print(f'Relative error: {abs(dif_loss - estimated_dif_loss) / abs(estimated_dif_loss)}')
