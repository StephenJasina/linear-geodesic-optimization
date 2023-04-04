import numpy as np
from scipy import linalg

from linear_geodesic_optimization.mesh.sphere import Mesh as SphereMesh
from linear_geodesic_optimization.optimization import geodesic, laplacian

frequency = 20
M = SphereMesh(frequency)

laplacian_forward = laplacian.Forward(M)
geodesic_forward = geodesic.Forward(M, laplacian_forward)

geodesic_forward.calc(M.nearest_vertex_index(
    SphereMesh.longitude_latitude_to_direction(0, 0)))
phi = geodesic_forward.phi

rng = np.random.default_rng()
for _ in range(10):
    direction = np.array([rng.random(), rng.random(), rng.random()])
    direction = direction / linalg.norm(direction)
    print(f'Estimated: {phi[M.nearest_vertex_index(direction)]:.6f};',
        end=' ')
    print(f'True: {np.arccos(direction[0]):.6f}')
