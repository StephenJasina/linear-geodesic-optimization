import json
import os

from matplotlib import pyplot as plt
import numpy as np

from linear_geodesic_optimization import data
from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh
from linear_geodesic_optimization.plot import get_heat_map

width = 40
height = 40
mesh = RectangleMesh(width, height)

cutoff = 16
network_name = f'Graph U.S. ({cutoff})'
data_file_name = os.path.join('graph_US', f'graph{cutoff}.graphml')
data_file_path = os.path.join('..', 'data', data_file_name)
data_name, data_type = os.path.splitext(os.path.basename(data_file_name))

if data_type == '.json':
    coordinates, network_edges, network_curvatures, network_latencies = data.read_json(data_file_path)
elif data_type == '.graphml':
    coordinates, network_edges, network_curvatures, network_latencies = data.read_graphml(data_file_path)
network_vertices = mesh.map_coordinates_to_support(coordinates)

mesh.set_parameters(np.random.random(mesh.get_parameters().shape))
vertices = mesh.get_vertices()
x = list(sorted(set(vertices[:,0])))
y = list(sorted(set(vertices[:,1])))
z = vertices[:,2].reshape(len(x), len(y))

width = len(x)
height = len(y)

x = x[1:width - 1]
y = y[1:height - 1]
z = z[1:width - 1,1:height - 1]
z = z - np.amin(z)

heat_map = get_heat_map(x, y, None, network_name,
                        network_vertices, network_edges, network_curvatures, network_vertices)
heat_map.savefig('network.png', dpi=300)
plt.show()
