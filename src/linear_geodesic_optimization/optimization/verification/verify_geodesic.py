import numpy as np

from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh
from linear_geodesic_optimization.optimization import geodesic, laplacian

mesh = RectangleMesh(10, 10)
z = np.random.rand(100)
mesh.set_parameters(z)

l = 37
gamma = [83]

laplacian_forward = laplacian.Forward(mesh)
laplacian_reverse = laplacian.Reverse(mesh, laplacian_forward)
geodesic_forward = geodesic.Forward(mesh, laplacian_forward)
geodesic_reverse = geodesic.Reverse(mesh, geodesic_forward, laplacian_reverse)

geodesic_forward.calc(gamma)
phi_0 = geodesic_forward.phi

geodesic_reverse.calc(gamma, mesh.get_partials()[l], l)
dif_phi = geodesic_reverse.dif_phi

# Can't be too much smaller than 1e-5 or we get underflow
delta = 1e-5
z[l] += delta
mesh.set_parameters(z)

geodesic_forward.calc(gamma)
phi_delta = geodesic_forward.phi

approx_dif_phi = np.array(phi_delta - phi_0) / delta

# Check derivative is close
print(np.max(np.abs(approx_dif_phi - dif_phi)))
