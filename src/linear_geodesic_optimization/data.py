import json
import os

import networkx as nx

def read_json(data_file_path):
    coordinates = []
    label_to_index = {}
    network_edges = []
    network_curvatures = []
    network_latencies = []

    with open(data_file_path) as f:
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

def read_graphml(data_file_path):
    network = nx.read_graphml(data_file_path)
    coordinates = [(node['lat'], node['long']) for node in network.nodes.values()]
    label_to_index = {label: index for index, label in enumerate(network.nodes)}
    network_edges = [(label_to_index[u], label_to_index[v]) for u, v in network.edges]
    network_curvatures = [edge['ricciCurvature'] for edge in network.edges.values()]
    network_latencies = [[] for _ in coordinates]

    return coordinates, network_edges, network_curvatures, network_latencies

def map_latencies_to_mesh(mesh, network_vertices, network_latencies):
    latencies = {
        mesh.nearest_vertex_index(network_vertices[i]): [
            (mesh.nearest_vertex_index(network_vertices[j]), latency)
            for j, latency in j_latency_pairs
        ]
        for i, j_latency_pairs in enumerate(network_latencies)
    }

    return latencies
