import sys

import dcelmesh
import numpy as np

sys.path.append('.')
from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh
from linear_geodesic_optimization.optimization.geodesic \
    import Computer as Geodesic
from linear_geodesic_optimization.optimization.laplacian \
    import Computer as Laplacian


width = 15
height = 15
mesh = RectangleMesh(width, height)
np.random.seed(0)
z = np.random.random(width * height)
z = mesh.set_parameters(z)

laplacian = Laplacian(mesh)
geodesic = Geodesic(mesh, 0, width * height - 1, laplacian)

geodesic.forward()
print(geodesic.distance)
for element in geodesic.path:
    if isinstance(element, dcelmesh.Mesh.Vertex):
        print(f'{element.index()}')
    else:
        print(f'({element.origin().index()}, {element.destination().index()})')
