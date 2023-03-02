import json
import os

from matplotlib import pyplot as plt
import numpy as np

from linear_geodesic_optimization.mesh.adaptive import Mesh as AdaptiveMesh
from linear_geodesic_optimization.plot import get_heat_map, get_mesh_plot

network_name = 'Elbow'
data_directory = os.path.join('..', 'data', 'elbow')

coordinates = None
label_to_index = {}
with open(os.path.join(data_directory, 'position.json')) as f:
    position_json = json.load(f)

    label_to_index = {label: index
                      for index, label in enumerate(position_json)}

    coordinates = [None for _ in range(len(position_json))]
    for vertex, position in position_json.items():
        coordinates[label_to_index[vertex]] = position

network_vertices = AdaptiveMesh.map_coordinates_to_support(coordinates)

network_curvatures = []
with open(os.path.join(data_directory, 'curvature.json')) as f:
    curvature_json = json.load(f)

    for edge, curvature in curvature_json.items():
        i = label_to_index[edge[0]]
        j = label_to_index[edge[1]]

        network_curvatures.append(((i, j), curvature))

points = set(tuple(network_vertex) for network_vertex in network_vertices)
for (i, j), _ in network_curvatures:
    for p in np.linspace(network_vertices[i], network_vertices[j], 10):
        points.add(tuple(p))
points = list(points)

mesh = AdaptiveMesh(AdaptiveMesh.map_coordinates_to_support(points), 0.1)
z = np.array([
    (4**2 - x**2 - y**2)**0.5
    for x, y, _ in mesh.get_vertices()
])
z = z - np.amin(z)
z = mesh.set_parameters(z)
get_heat_map(network_vertices=list(mesh.get_vertices()),
             network_curvatures=[((i, j), 1.) for i, js in enumerate(mesh.get_edges()) for j in js],
             extra_points=points)
# get_mesh_plot(mesh, 'Adaptive Mesh Test', remove_boundary=False)
plt.show()
