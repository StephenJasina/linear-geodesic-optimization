import csv
import json
import os

import networkx as nx
import numpy as np

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

def mercator(longitude, latitude):
    '''
    Given a longitude in [-180, 180] and a latitude in [-90, 90], return an
    (x, y) pair representing the location on a Mercator projection. Assuming
    the latitude is no larger/smaller than +/- 85 (approximately), the pair
    will lie in [-0.5, 0.5]^2.
    '''

    x = longitude / 360.
    y = np.log(np.tan(np.pi / 4. + latitude * np.pi / 360.)) / (2. * np.pi)
    return (x, y)

def inverse_mercator(x, y):
    longitude = x * 360.
    latitude = np.arctan(np.exp(y * 2. * np.pi)) * 360. / np.pi - 90.
    return (longitude, latitude)

def read_graphml(data_file_path, latencies_file_path=None):
    network = nx.read_graphml(data_file_path)
    coordinates = [mercator(node['long'], node['lat']) for node in network.nodes.values()]
    label_to_index = {label: index for index, label in enumerate(network.nodes)}
    network_edges = [(label_to_index[u], label_to_index[v]) for u, v in network.edges]
    network_curvatures = [edge['ricciCurvature'] for edge in network.edges.values()]
    network_latencies = [[] for _ in coordinates]
    if latencies_file_path is not None:
        with open(latencies_file_path) as latencies_file:
            latencies_reader = csv.reader(latencies_file)
            for row in latencies_reader:
                latency = float(row[2])
                if latency != 0.:
                    network_latencies[label_to_index[row[0]]].append(
                        (label_to_index[row[1]], latency)
                    )

    return coordinates, network_edges, network_curvatures, network_latencies

def map_latencies_to_mesh(mesh, network_vertices, network_latencies):
    latencies = {}

    # Can't do a dict comprehension since multiple vertices could map to the
    # same mesh point
    for i, j_latency_pairs in enumerate(network_latencies):
        key = mesh.nearest_vertex_index(network_vertices[i])
        if key not in latencies:
            latencies[key] = []

        latencies[key] += [
            (mesh.nearest_vertex_index(network_vertices[j]), latency)
            for j, latency in j_latency_pairs
        ]

    return latencies
