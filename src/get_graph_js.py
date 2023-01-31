import json
import os

from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh

if __name__ == '__main__':
    toy_directory = os.path.join('..', 'data', 'two_islands')

    # These two parameters shouldn't actually matter, but are required
    # to instantiate a RectangleMesh object
    width = 20
    height = 20

    mesh = RectangleMesh(width, height)

    with open(os.path.join(toy_directory, 'position.json')) as f:
        position_json = json.load(f)

        label_to_index = {label: index
                          for index, label in enumerate(position_json)}

        coordinates = [None for _ in range(len(position_json))]
        for vertex, position in position_json.items():
            coordinates[label_to_index[vertex]] = position

    network_vertices = mesh.map_coordinates_to_support(coordinates)

    for index, label in enumerate(position_json):
        position = network_vertices[index]
        print(f'addVertex(null, {position[0] * 10.}, {position[1] * 10.}, true, name="{label}");')

    print()

    with open(os.path.join(toy_directory, 'ricci_curvature.json')) as f:
        curvature_json = json.load(f)

        for edge, curvature in curvature_json.items():
            u = label_to_index[edge[0]]
            v = label_to_index[edge[1]]

            print(f'addEdge(null, {u}, {v}, {curvature});')
