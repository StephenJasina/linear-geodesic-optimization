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

laplacian_forward = laplacian.Forward(mesh)
curvature_forward = curvature.Forward(mesh, laplacian_forward)
smooth_forward = smooth.Forward(mesh, laplacian_forward, curvature_forward)
laplacian_reverse = laplacian.Reverse(mesh, laplacian_forward)
curvature_reverse = curvature.Reverse(mesh, laplacian_forward,
                                      curvature_forward, laplacian_reverse)
smooth_reverse = smooth.Reverse(mesh, laplacian_forward, curvature_forward,
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
