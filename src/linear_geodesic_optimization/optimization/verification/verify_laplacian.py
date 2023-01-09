import numpy as np

from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh
from linear_geodesic_optimization.optimization import laplacian

mesh = RectangleMesh(10, 10)
z = np.random.rand(100)
mesh.set_parameters(z)
l=37

laplacian_forward = laplacian.Forward(mesh)
laplacian_reverse = laplacian.Reverse(mesh, laplacian_forward)

laplacian_forward.calc()
LC_0 = laplacian_forward.LC_dirichlet

laplacian_reverse.calc(mesh.get_partials()[l], l)
dif_LC = laplacian_reverse.dif_LC_dirichlet

# Can't be too much smaller than 1e-5 or we get underflow
delta = 1e-5
z[l] += delta
mesh.set_parameters(z)

laplacian_forward.calc()
LC_delta = laplacian_forward.LC_dirichlet

approx_dif_LC = np.array(LC_delta - LC_0) / delta

# Check derivative is close
print(np.max(np.abs(approx_dif_LC - dif_LC)))

print(np.linalg.norm(LC_0.toarray() - LC_0.toarray().T))
