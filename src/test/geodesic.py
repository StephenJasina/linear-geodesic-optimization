import sys

import numpy as np

sys.path.append('.')
from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh
from linear_geodesic_optimization.optimization.geodesic \
    import Computer as Geodesic
from linear_geodesic_optimization.optimization.laplacian \
    import Computer as Laplacian


width = 30
height = 30
mesh = RectangleMesh(width, height)
# z = np.array([
#     -(16.**2
#         - (i / (width - 1) - 0.5)**2
#         - (j / (height - 1) - 0.5)**2)**0.5
#     for i in range(width)
#     for j in range(height)
# ])
# np.random.seed(0)
# z = np.random.random(width * height)
z = np.zeros(width * height)
z = mesh.set_parameters(z)

laplacian = Laplacian(mesh)
geodesic = Geodesic(mesh, 0, width * height - 1, laplacian)

geodesic.forward()
print(geodesic.distance)
print(geodesic.path_edges)
