import numpy as np

from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh
from linear_geodesic_optimization.optimization import laplacian, smooth

mesh = RectangleMesh(10, 10)
z = np.random.rand(100)
mesh.set_parameters(z)
l=37

laplacian_forward = laplacian.Forward(mesh)
smooth_forward = smooth.Forward(mesh, laplacian_forward)
laplacian_reverse = laplacian.Reverse(mesh, laplacian_forward)
smooth_reverse = smooth.Reverse(mesh, laplacian_forward, laplacian_reverse)

smooth_forward.calc()
L_smooth_0 = smooth_forward.L_smooth

smooth_reverse.calc(mesh.get_partials()[l], l)
dif_L_smooth = smooth_reverse.dif_L_smooth

# Can't be too much smaller than 1e-5 or we get underflow
delta = 1e-5
z[l] += delta
mesh.set_parameters(z)

smooth_forward.calc()
L_smooth_delta = smooth_forward.L_smooth

approx_dif_L_smooth = np.array(L_smooth_delta - L_smooth_0) / delta

# Check derivative is close
print(abs(approx_dif_L_smooth - dif_L_smooth))
print(approx_dif_L_smooth / dif_L_smooth)
