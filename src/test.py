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

mesh = AdaptiveMesh(AdaptiveMesh.map_coordinates_to_support(network_vertices), 0.05)
get_heat_map(network_vertices=list(mesh.get_vertices()) + network_vertices,
             network_curvatures=[((i, j), 1.) for i, js in enumerate(mesh.get_edges()) for j in js])
plt.show()
