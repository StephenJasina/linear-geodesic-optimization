import numpy as np

from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh
from linear_geodesic_optimization.optimization import geodesic, laplacian

from linear_geodesic_optimization.plot import get_heat_map

mesh = RectangleMesh(41, 41)
z = np.zeros(1681)
mesh.set_parameters(z)

gamma = [840]

laplacian_forward = laplacian.Forward(mesh)
geodesic_forward = geodesic.Forward(mesh, laplacian_forward)

geodesic_forward.calc(gamma)
phi = geodesic_forward.phi

vertices = mesh.get_vertices()
x = list(sorted(set(vertices[:,0])))
y = list(sorted(set(vertices[:,1])))
z = phi.reshape(len(x), len(y), order='F')
get_heat_map(x, y, z).show()
