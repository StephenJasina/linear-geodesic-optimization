import json
import os

from linear_geodesic_optimization import data
from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh

if __name__ == '__main__':
    width = 40
    height = 40
    mesh = RectangleMesh(width, height)

    network_name = 'Graph U.S. (16)'
    data_file_name = 'graph_US_16.graphml'
    data_file_path = os.path.join('..', 'data', data_file_name)
    data_name, data_type = os.path.splitext(os.path.basename(data_file_name))

    if data_type == '.json':
        coordinates, network_edges, network_curvatures, network_latencies = data.read_json(data_file_path)
    elif data_type == '.graphml':
        coordinates, network_edges, network_curvatures, network_latencies = data.read_graphml(data_file_path)
    network_vertices = mesh.map_coordinates_to_support(coordinates)

    for index, label in enumerate(network_vertices):
        position = network_vertices[index]
        print(f'  addVertex(null, {position[0] * 20.}, {position[1] * -20.}, true, name="{str(index)}");')

    print()

    for edge, curvature in zip(network_edges, network_curvatures):
        print(f'  addEdge(null, {str(edge[0])}, {str(edge[1])}, {curvature});')
