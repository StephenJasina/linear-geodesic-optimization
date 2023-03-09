import json
import os

from matplotlib import pyplot as plt
import numpy as np

from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh
from linear_geodesic_optimization.plot import get_heat_map

width = 40
height = 40
mesh = RectangleMesh(width, height)

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

network_vertices = mesh.map_coordinates_to_support(coordinates)

network_edges = []
network_curvatures = []
with open(os.path.join(data_directory, 'curvature.json')) as f:
    curvature_json = json.load(f)

    for edge, curvature in curvature_json.items():
        u = label_to_index[edge[0]]
        v = label_to_index[edge[1]]

        network_edges.append((u, v))
        network_curvatures.append(curvature)

mesh.set_parameters(np.random.random(mesh.get_parameters().shape))
vertices = mesh.get_vertices()
x = list(sorted(set(vertices[:,0])))
y = list(sorted(set(vertices[:,1])))
z = vertices[:,2].reshape(len(x), len(y), order='F')

width = len(x)
height = len(y)

x = x[1:width - 1]
y = y[1:height - 1]
z = z[1:width - 1,1:height - 1]
z = z - np.amin(z)

get_heat_map(x, y, None, network_name,
             network_vertices, network_edges, network_curvatures)
plt.show()
