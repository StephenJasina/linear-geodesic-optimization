import csv
import json
import os

from matplotlib import pyplot as plt
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

def read_graphml(data_file_path, latencies_file_path=None):
    network = nx.read_graphml(data_file_path)
    coordinates = [(node['lat'], node['long']) for node in network.nodes.values()]
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
    latencies = {
        mesh.nearest_vertex_index(network_vertices[i]): [
            (mesh.nearest_vertex_index(network_vertices[j]), latency)
            for j, latency in j_latency_pairs
        ]
        for i, j_latency_pairs in enumerate(network_latencies)
    }

    return latencies

def compute_convex_hull(points):
    '''
    An implementation of the Graham scan algorithm. As input, take a list of
    pairs. Returns a list of the indices of the vertices on the convex hull,
    oriented counter-clockwise.
    '''

    points = np.array(points)

    pivot_point_index = 0
    pivot_point = points[pivot_point_index]
    for index, point in enumerate(points):
        if point[1] < pivot_point[1] or (point[1] == pivot_point[1] and point[0] < pivot_point[0]):
            pivot_point_index = index
            pivot_point = point

    points = points - pivot_point

    sorted_indices = [index for _, index in sorted(
        (
            (
                -point[0] / np.linalg.norm(point),
                np.linalg.norm(point)
            ),
            index
        )
        for index, point in enumerate(points)
        if point @ point > 0
    )]

    convex_hull = [pivot_point_index]
    for index in sorted_indices:
        point_index = points[index]
        print(index, point_index + pivot_point)
        while True:
            if len(convex_hull) <= 1:
                break

            point_top = points[convex_hull[-1]]
            point_second_to_top = points[convex_hull[-2]]

            if np.cross(point_top - point_index, point_second_to_top - point_index) < 0:
                break

            del(convex_hull[-1])

        convex_hull.append(index)

    return convex_hull
