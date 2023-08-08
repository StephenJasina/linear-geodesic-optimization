import json
import os

from matplotlib import pyplot as plt
import numpy as np

from linear_geodesic_optimization import data
from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh
from linear_geodesic_optimization.plot import get_heat_map

width = 50
height = 50
mesh = RectangleMesh(width, height)

# cutoff = 16
# network_name = f'Graph U.S. ({cutoff})'
# data_file_name = os.path.join('graph_US', f'graph{cutoff}.graphml')
network_name = 'Two Islands'
data_file_name = os.path.join('toy', 'two_islands.graphml')
data_file_path = os.path.join('..', 'data', data_file_name)
data_name, data_type = os.path.splitext(os.path.basename(data_file_name))

coordinates, network_edges, network_curvatures, network_latencies = data.read_graphml(data_file_path)
coordinates = np.array(coordinates)
network_vertices = mesh.map_coordinates_to_support(coordinates, 0.8)

mesh.set_parameters(np.random.random(mesh.get_parameters().shape))
vertices = mesh.get_coordinates()
x = list(sorted(set(vertices[:,0])))
y = list(sorted(set(vertices[:,1])))
z = vertices[:,2].reshape(len(x), len(y))

width = len(x)
height = len(y)

z = z - np.amin(z)

# network_curvatures = [None] * len(network_curvatures)
heat_map = get_heat_map(x, y, None, network_name,
                        network_vertices, network_edges, network_curvatures, network_vertices)
heat_map.savefig('network.png', dpi=1000)
plt.show()
