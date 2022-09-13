import numpy as np

from linear_geodesic_optimization.mesh.sphere import Mesh as Sphere
from linear_geodesic_optimization.plot import Animation3D

def on_fat_edge(e, r, epsilon):
    u, v = e

    ru = r @ u
    rv = r @ v
    uv = u @ v
    cos_eps = np.cos(epsilon)

    c = np.sqrt((ru**2 + rv**2 - 2* ru * rv * uv) / (1 - uv**2))

    return max(ru, rv) > cos_eps or (c > cos_eps and min(ru, rv) > c * uv)

def get_fat_edge(vertices, e, epsilon):
    fat_edge = []
    for i in range(vertices.shape[0]):
        if on_fat_edge(e, vertices[i,:], epsilon):
            fat_edge.append(i)
    return fat_edge

mesh = Sphere(10)
rho = mesh.get_parameters()
vertices = mesh.get_vertices()
u = np.random.rand(3)
u = u / np.linalg.norm(u)
v = np.random.rand(3)
v = v / np.linalg.norm(v)

fat_edge = get_fat_edge(vertices, (u, v), 0.1)
for i in fat_edge:
    rho[i] *= 2
rho[mesh.nearest_vertex_index(u)] *= 2
rho[mesh.nearest_vertex_index(v)] *= 2
mesh.set_parameters(rho)

animation3d = Animation3D()
animation3d.add_frame(mesh)
animation3d.get_fig().show()
