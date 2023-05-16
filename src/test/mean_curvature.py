# TODO: Update this to reflect the rewrite

import numpy as np

from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh
from linear_geodesic_optimization.optimization import curvature

width = 10
height = 10
mesh = RectangleMesh(width, height)
z = np.array([
    -(16.**2
        - (i / (width - 1) - 0.5)**2
        - (j / (height - 1) - 0.5)**2)**0.5
    for i in range(width)
    for j in range(height)
])
np.random.seed(0)
z = np.random.random(width * height)
z = mesh.set_parameters(z)

curvature_forward = curvature.Forward(mesh)
curvature_forward.calc()
# for i in range(10):
#     for j in range(10):
#         print(f'{curvature_forward.kappa_H[height * j + (height - i - 1)]: 8.4f}', end=' ')
#     print()

for i in range(10):
    for j in range(10):
        kappa = curvature_forward.kappa_H[height * j + (height - i - 1)]
        print(f'{kappa: 10.6f}', end=' ')
    print()

# curvature_reverse = curvature.Reverse(mesh)

# curvature_reverse.calc(mesh.get_partials()[9], 9)

# for i in range(10):
#     for j in range(10):
#         print(f'{curvature_reverse.dif_kappa_H[height * j + (height - i - 1)]: 8.4f}', end=' ')
#     print()
