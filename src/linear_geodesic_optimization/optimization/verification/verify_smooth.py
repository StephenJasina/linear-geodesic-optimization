import json
import os

import numpy as np

from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh
from linear_geodesic_optimization.optimization import laplacian, curvature, smooth

toy_directory = os.path.join('..', 'data', 'toy')

# Construct a mesh
width = 10
height = 10
mesh = RectangleMesh(width, height)
z = np.random.rand(100)
mesh.set_parameters(z)
l=37

coordinates = None
label_to_index = {}
with open(os.path.join(toy_directory, 'position.json')) as f:
    position_json = json.load(f)

    label_to_index = {label: index for index, label in enumerate(position_json)}

    coordinates = [None for _ in range(len(position_json))]
    for vertex, position in position_json.items():
        coordinates[label_to_index[vertex]] = position

network_vertices = mesh.map_coordinates_to_support(coordinates)

network_edges = []
ts = {i: [] for i in range(len(network_vertices))}
with open(os.path.join(toy_directory, 'latency.json')) as f:
    latency_json = json.load(f)

    for edge, latency in latency_json.items():
        u = label_to_index[edge[0]]
        v = label_to_index[edge[1]]

        network_edges.append((u, v))

        ts[u].append((v, latency))

ricci_curvatures = []
with open(os.path.join(toy_directory, 'ricci_curvature.json')) as f:
    ricci_curvatures = list(json.load(f).values())

laplacian_forward = laplacian.Forward(mesh)
curvature_forward = curvature.Forward(mesh, network_vertices,
                                      network_edges, ricci_curvatures,
                                      mesh.get_epsilon(),
                                      laplacian_forward)
smooth_forward = smooth.Forward(mesh, network_vertices,
                                network_edges, ricci_curvatures,
                                mesh.get_epsilon(),
                                laplacian_forward, curvature_forward)
laplacian_reverse = laplacian.Reverse(mesh, laplacian_forward)
curvature_reverse = curvature.Reverse(mesh, network_vertices,
                                      network_edges, ricci_curvatures,
                                      mesh.get_epsilon(),
                                      laplacian_forward, curvature_forward,
                                      laplacian_reverse)
smooth_reverse = smooth.Reverse(mesh, network_vertices,
                                network_edges, ricci_curvatures,
                                mesh.get_epsilon(),
                                laplacian_forward, curvature_forward,
                                laplacian_reverse, curvature_reverse)

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

# initial_radius = 1
# def get_z(x, y):
#     return (initial_radius**2 - x**2 - y**2)**0.5

# z = np.array([
#     get_z(i / (width - 1) - 0.5, j / (height - 1) - 0.5)
#     for i in range(width)
#     for j in range(height)
# ]).reshape((width * height,))
# z = mesh.set_parameters(z - min(z))

# laplacian_forward.calc()
# LC = laplacian_forward.LC_dirichlet
# curvature_forward.calc()
# kappa = curvature_forward.kappa

# # center = [i for i in range(len(mesh.get_vertices())) if i not in mesh.get_boundary_vertices()]
# # print(len(center))
# # LC = LC[center][:,center]
# # kappa = kappa[center]

# print(-kappa.T @ LC @ kappa)
