import json
import os

import numpy as np

from linear_geodesic_optimization import data
from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh

if __name__ == '__main__':
    width = 50
    height = 50
    mesh = RectangleMesh(width, height)

    scale_factor = 0.8

    # cutoff = 22
    # data_file_name = os.path.join('graph_US', f'graph{cutoff}.graphml')
    data_file_name = os.path.join('toy', 'two_islands_complete.graphml')
    data_file_path = os.path.join('..', 'data', data_file_name)
    data_name, _ = os.path.splitext(os.path.basename(data_file_name))

    coordinates, network_edges, network_curvatures, network_latencies = data.read_graphml(data_file_path)
    coordinates = np.array(coordinates)
    network_vertices = mesh.map_coordinates_to_support(coordinates, scale_factor)

    print('Edit the generateGraph and createMap functions in script.js')
    print('\n' + '=' * 80 + '\n')

    for index, label in enumerate(network_vertices):
        position = network_vertices[index]
        # addVertex accepts (latitude, longitude) pairs, so we have to swap x and y
        print(f'  addVertex(null, {position[1] * 20.}, {position[0] * 20.}, true, name="{str(index)}");')

    print()

    for edge, curvature in zip(network_edges, network_curvatures):
        print(f'  addEdge(null, {str(edge[0])}, {str(edge[1])}, {curvature});')

    print('\n' + '=' * 80 + '\n')

    coordinates = np.array(coordinates)
    center_xy = (np.amin(coordinates, axis=0) + np.amax(coordinates, axis=0)) / 2.
    center = data.inverse_mercator(*center_xy)
    left, _ = data.inverse_mercator(np.amin(coordinates[:,0]), 0)
    right, _ = data.inverse_mercator(np.amax(coordinates[:,0]), 0)
    zoom_factor = scale_factor * 360. / (right - left)

    print(f'Center is {center}')
    print(f'Zoom factor is {zoom_factor}')
