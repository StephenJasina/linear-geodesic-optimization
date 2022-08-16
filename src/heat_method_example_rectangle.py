import numpy as np

from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh
from linear_geodesic_optimization.optimization import geodesic, laplacian

width = 64
height = 64
M = RectangleMesh(width, height)

laplacian_forward = laplacian.Forward(M)
geodesic_forward = geodesic.Forward(M, laplacian_forward)

geodesic_forward.calc(M.nearest_vertex_index(0., 0.))
phi = geodesic_forward.phi

rng = np.random.default_rng()
for _ in range(10):
    x = rng.random()
    y = rng.random()
    print(f'Estimated: {phi[M.nearest_vertex_index(x, y)]:.6f};',
        end=' ')
    print(f'True: {np.sqrt(x**2 + y**2):.6f}')
