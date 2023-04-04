import pickle

import numpy as np

from linear_geodesic_optimization import convex_hull
from linear_geodesic_optimization import data
from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh
from linear_geodesic_optimization.mesh.sphere import Mesh as SphereMesh
from linear_geodesic_optimization.optimization import geodesic, linear_regression

mesh_path = '../out_US/graph_US_16/mean/0.0_1.0_0.002_16.0_40_40/500'
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
t = []
for i, j_latency_pairs in latencies.items():
    geodesic_forward.calc(i)
    for j, latency in j_latency_pairs:
        phi.append(geodesic_forward.phi[j])
        t.append(latency)

phi = np.array(phi)
t = np.array(t)

linear_regression_forward = linear_regression.Forward()

print("Using geodesics")
linear_regression_forward.calc(phi, t)
print(linear_regression_forward.lse)
print(linear_regression_forward.get_beta(phi, t))

t = []
d = []
for i, j_latency_pairs in enumerate(network_latencies):
    u = SphereMesh.latitude_longitude_to_direction(*coordinates[i])
    for j, latency in j_latency_pairs:
        v = SphereMesh.latitude_longitude_to_direction(*coordinates[j])
        t.append(latency)
        d.append(np.arccos(max(min(u @ v, 1.), -1.)))

d = np.array(d)
t = np.array(t)

print("Using great circle")
linear_regression_forward.calc(d, t)
print(linear_regression_forward.lse)
print(linear_regression_forward.get_beta(d, t))
