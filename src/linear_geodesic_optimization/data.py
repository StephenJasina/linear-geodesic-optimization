import json
import os

def read_json(data_file):
    coordinates = []
    label_to_index = {}
    network_edges = []
    network_curvatures = []
    network_latencies = []

    with open(data_file) as f:
        full_json = json.load(f)
        position_json = full_json['position']
        curvature_json = full_json['curvature']
        latency_json = full_json['latency']

        label_to_index = {label: index for index, label in enumerate(position_json)}
        coordinates = list(position_json.values())

        network_curvatures = list(curvature_json.values())
        network_edges = [
            (label_to_index[edge[0]], label_to_index[edge[1]])
            for edge in curvature_json
        ]

        network_latencies = [[] for _ in coordinates]
        for edge, latency in latency_json.items():
            network_latencies[label_to_index[edge[0]]].append(
                (label_to_index[edge[1]], latency)
            )

    return coordinates, network_edges, network_curvatures, network_latencies

def read_graphml(data_directory):
    pass

def map_latencies_to_mesh(mesh, network_vertices, network_latencies):
    latencies = {
        mesh.nearest_vertex_index(network_vertices[i]): [
            (mesh.nearest_vertex_index(network_vertices[j]), latency)
            for j, latency in j_latency_pairs
        ]
        for i, j_latency_pairs in enumerate(network_latencies)
    }

    return latencies
