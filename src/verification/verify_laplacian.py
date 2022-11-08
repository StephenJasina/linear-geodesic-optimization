import numpy as np

from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh
from linear_geodesic_optimization.optimization import laplacian

mesh = RectangleMesh(10, 10)
z = np.random.rand(100)
mesh.set_parameters(z)

laplacian_forward = laplacian.Forward(mesh)
laplacian_reverse = laplacian.Reverse(mesh, laplacian_forward)

laplacian_forward.calc()
L_0 = laplacian_forward.L

laplacian_reverse.calc(mesh.get_partials()[37], 37)
dif_L = laplacian_reverse.dif_L

# Can't be too much smaller than 1e-5 or we get underflow
delta = 1e-5
z[37] += delta
mesh.set_parameters(z)

laplacian_forward.calc()
L_delta = laplacian_forward.L

approx_dif_L = np.array(L_delta - L_0) / delta

# Check derivative is close
print(np.max(np.abs(approx_dif_L - dif_L)))
