import csv
import os
import pickle
import sys

from matplotlib import pyplot as plt
import numpy as np

from linear_geodesic_optimization import convex_hull
from linear_geodesic_optimization.data import input_network, input_mesh
from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh
from linear_geodesic_optimization.optimization.geodesic import Computer as Geodesic

max_iterations = 10000
use_postprocessing = False

def get_geodesics(mesh, network_coordinates, bounding_box, network_edges,
                  labels, compute_all_vertex_pairs=False):
    network_vertices = mesh.map_coordinates_to_support(
        np.array(network_coordinates), np.float64(0.8), bounding_box)
    nearest_vertex_indices = [mesh.nearest_vertex(network_vertex).index()
                              for network_vertex in network_vertices]

    valid_edges = network_edges
    if compute_all_vertex_pairs:
        valid_edges = [(i, j)
                       for i in range(len(network_vertices))
                       for j in range(len(network_vertices))]
    valid_pairs = set([
        (nearest_vertex_indices[i], nearest_vertex_indices[j])
        for i, j in valid_edges
    ])

    # Don't compute geodesics for the same pair of mesh vertices twice
    geodesics_unique = {}
    for i, j in valid_edges:
        mesh_i = nearest_vertex_indices[i]
        mesh_j = nearest_vertex_indices[j]

        geodesic_forward = Geodesic(mesh, mesh_i, mesh_j)
        geodesic_forward.forward()
        geodesics_unique[mesh_i,mesh_j] = geodesic_forward.distance
        geodesics_unique[mesh_j,mesh_i] = geodesic_forward.distance

    geodesics = []
    for i, j in valid_edges:
        mesh_i = nearest_vertex_indices[i]
        mesh_j = nearest_vertex_indices[j]

        label_i = labels[i]
        label_j = labels[j]

        geodesics.append(((label_i,label_j), geodesics_unique[mesh_i,mesh_j]))

    return geodesics

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python3 <directory name>')
        sys.exit(0)

    directory = sys.argv[1]

    if not os.path.exists(os.path.join(directory, 'parameters')):
        print('Error: supplied directory must contain file named "parameters"')
        sys.exit(0)

    with open(os.path.join(directory, 'parameters'), 'rb') as f:
        parameters = pickle.load(f)

        width = parameters['width']
        height = parameters['height']

    with open(os.path.join(directory, '0'), 'rb') as f:
        iteration_data = pickle.load(f)
        z_0 = np.array(iteration_data['mesh_parameters'])

    iteration = min(
        max_iterations,
        max(int(name) for name in os.listdir(directory) if name.isdigit())
    )
    with open(os.path.join(directory, str(iteration)), 'rb') as f:
        iteration_data = pickle.load(f)
        z = np.array(iteration_data['mesh_parameters'])

    probes_file_path = os.path.join('..', 'data', parameters['probes_filename'])
    latencies_file_path = os.path.join('..', 'data', parameters['latencies_filename'])
    epsilon = parameters['epsilon']
    clustering_distance = parameters['clustering_distance']
    should_remove_tivs = parameters['should_remove_TIVs']
    network, latencies = input_network.get_graph(
        probes_file_path, latencies_file_path,
        epsilon, clustering_distance,
        should_remove_tivs=should_remove_tivs,
        should_include_latencies=True
    )
    network = input_network.extract_from_graph(network, latencies, with_labels=True)
    network, (labels, _) = network[:-2], network[-2:]
    network_coordinates, bounding_box, network_edges, _, _ = network
    mesh = input_mesh.get_mesh(z, width, height, network, use_postprocessing, z_0)

    geodesics = get_geodesics(mesh, network_coordinates, bounding_box,
                              network_edges, labels, True)

    with open(os.path.join(directory, 'geodesics.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['source', 'destination', 'geodesic_distance'])
        for (a, b), geodesic in geodesics:
            writer.writerow([a, b, geodesic])
