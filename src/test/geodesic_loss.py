import sys
import time

import dcelmesh
import numpy as np

sys.path.append('.')
from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh
from linear_geodesic_optimization.optimization.geodesic \
    import Computer as Geodesic
from linear_geodesic_optimization.optimization.geodesic_loss \
    import Computer as GeodesicLoss


width = 30
height = 30
n = 10

seed = time.time_ns()
seed = seed % (2**32 - 1)
np.random.seed(seed)
print(f'Seed: {seed}')

mesh = RectangleMesh(width, height, extent=1.)
geodesics = [Geodesic(mesh,
                      np.random.randint(width * height),
                      np.random.randint(width * height))
             for _ in range(n)]
geodesic_loss = GeodesicLoss(geodesics,
                             (np.random.random(n) - 0.5) * 2.)

z = mesh.set_parameters(np.random.random(width * height))
dz = np.random.random(width * height)
dz = dz / np.linalg.norm(dz)
h = 1e-7

# Compute the partial derivative in the direction of offset
geodesic_loss.reverse()
dif_loss = np.float64(0.)
for j, d in geodesic_loss.dif_loss.items():
    dif_loss += d * dz[j]

# Estimate the partial derivative by adding, evaluating, and subtracting
mesh.set_parameters(z + h * dz)
geodesic_loss.forward()
loss_z_plus_dz = geodesic_loss.loss
mesh.set_parameters(z - h * dz)
geodesic_loss.forward()
loss_z_minus_dz = geodesic_loss.loss
estimated_dif_loss = (loss_z_plus_dz - loss_z_minus_dz) / (2. * h)

# Should print something close to 1
print(f'Quotient: {dif_loss / estimated_dif_loss:.6f}')
