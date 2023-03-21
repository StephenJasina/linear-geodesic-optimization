import json
import os

def read_json(data_directory):
    coordinates = None
    label_to_index = {}
    with open(os.path.join(data_directory, 'position.json')) as f:
        position_json = json.load(f)

        label_to_index = {label: index for index, label in enumerate(position_json)}

        coordinates = [None for _ in range(len(position_json))]
        for vertex, position in position_json.items():
            coordinates[label_to_index[vertex]] = position

    network_edges = []
    network_curvatures = []
    with open(os.path.join(data_directory, 'curvature.json')) as f:
        curvature_json = json.load(f)
        network_curvatures = list(curvature_json.values())
        network_edges = [
            (label_to_index[edge[0]], label_to_index[edge[1]])
            for edge in curvature_json
        ]

    network_latencies = []
    with open(os.path.join(data_directory, 'latency.json')) as f:
        latency_json = json.load(f)

        for edge, latency in latency_json.items():
            network_latencies[label_to_index[edge[0]]].append(
                (label_to_index[edge[1]], latency)
            )

    return coordinates, network_edges, network_curvatures, network_latencies

def read_graphml(data_directory):
    pass

def map_to_mesh(mesh, coordinates, network_latencies):
    network_vertices = mesh.map_coordinates_to_support(coordinates)
    latencies = {
        mesh.nearest_vertex_index(network_vertices[i]): [
            (mesh.nearest_vertex_index(network_vertices[j]), latency)
            for j, latency in j_latency_pairs
        ]
        for i, j_latency_pairs in network_latencies
    }

    return network_vertices, latencies