import pickle

import numpy as np

from linear_geodesic_optimization import convex_hull
from linear_geodesic_optimization import data
from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh
from linear_geodesic_optimization.mesh.sphere import Mesh as SphereMesh
from linear_geodesic_optimization.optimization import geodesic, linear_regression
from linear_geodesic_optimization import plot

mesh_path = '../out_US/graph_US_16/mean/0.0_1.0_0.002_16.0_40_40/0'
data_file_path = '../data/graph_US_16.graphml'
latencies_file_path = '../data/latencies_US.csv'

width = 40
height = 40
initial_radius = 16.

with open(mesh_path, 'rb') as f:
    z = pickle.load(f)['mesh_parameters']

z_0 = np.array([
    (initial_radius**2
        - (i / (width - 1) - 0.5)**2
        - (j / (height - 1) - 0.5)**2)**0.5
    for j in range(height)
    for i in range(width)
])

mesh = RectangleMesh(width, height)

coordinates, network_edges, network_curvatures, network_latencies = data.read_graphml(data_file_path, latencies_file_path)
network_vertices = mesh.map_coordinates_to_support(coordinates)
network_convex_hull = convex_hull.compute_convex_hull(network_vertices)
latencies = data.map_latencies_to_mesh(mesh, network_vertices, network_latencies)

vertices = mesh.get_vertices()
x = list(sorted(set(vertices[:,0])))
y = list(sorted(set(vertices[:,1])))
z = z - z_0
distances = np.array([
    np.linalg.norm(np.array([px, py]) - convex_hull.project_to_convex_hull([px, py], network_vertices, network_convex_hull))
    for py in y
    for px in x
])
z = (z - np.amin(z)) * np.exp(-100 * distances**2)
z = z - np.amin(z)
mesh.set_parameters(z)

geodesic_forward = geodesic.Forward(mesh)
geodesics = {}
phi = []
t_geodesic = []
for i, j_latency_pairs in latencies.items():
    geodesic_forward.calc(i)
    for j, latency in j_latency_pairs:
        phi.append(geodesic_forward.phi[j])
        t_geodesic.append(latency)

phi = np.array(phi)
t_geodesic = np.array(t_geodesic)

linear_regression_forward = linear_regression.Forward()

print('Using geodesics')
linear_regression_forward.calc(phi, t_geodesic)
beta_geodesic = linear_regression_forward.get_beta(phi, t_geodesic)
print(linear_regression_forward.lse)
print(beta_geodesic)

d_euclidean = []
t_euclidean = []
for i, j_latency_pairs in enumerate(network_latencies):
    u = np.array(coordinates[i])
    for j, latency in j_latency_pairs:
        v = np.array(coordinates[j])
        t_euclidean.append(latency)
        d_euclidean.append(np.linalg.norm(u - v))
d_euclidean = np.array(d_euclidean)
t_euclidean = np.array(t_euclidean)

print('Using Euclidean')
linear_regression_forward.calc(d_euclidean, t_euclidean)
beta_euclidean = linear_regression_forward.get_beta(d_euclidean, t_euclidean)
print(linear_regression_forward.lse)
print(beta_euclidean)

d_great_circle = []
t_great_circle = []
for i, j_latency_pairs in enumerate(network_latencies):
    u = SphereMesh.longitude_latitude_to_direction(*coordinates[i])
    for j, latency in j_latency_pairs:
        v = SphereMesh.longitude_latitude_to_direction(*coordinates[j])
        t_great_circle.append(latency)
        d_great_circle.append(np.arccos(max(min(u @ v, 1.), -1.)))
d_great_circle = np.array(d_great_circle)
t_great_circle = np.array(t_great_circle)

print('Using great circle')
linear_regression_forward.calc(d_great_circle, t_great_circle)
beta_great_circle = linear_regression_forward.get_beta(d_great_circle, t_great_circle)
print(linear_regression_forward.lse)
print(beta_great_circle)

from matplotlib import pyplot as plt
x = beta_geodesic[0] + beta_geodesic[1] * phi
# x = beta_great_circle[0] + beta_great_circle[1] * d_great_circle
# x = beta_euclidean[0] + beta_euclidean[1] * d_euclidean
y = t_geodesic
lim_min = min(min(x), min(y))
lim_max = max(max(x), max(y))

fig, ax = plt.subplots(1, 1)
ax.set_aspect('equal')
ax.plot(x, y, 'b.')
ax.set_xlim(lim_min, lim_max)
ax.set_ylim(lim_min, lim_max)
plt.show()
