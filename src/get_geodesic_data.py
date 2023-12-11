import csv
import os
import pickle
import sys

from matplotlib import pyplot as plt
import numpy as np

from linear_geodesic_optimization import convex_hull
from linear_geodesic_optimization import data
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
    network_convex_hulls = convex_hull.compute_connected_convex_hulls(
        network_vertices, network_edges)

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

        # Diagnostic information
        # label_i = labels[i]
        # label_j = labels[j]
        # print(label_i, label_j)

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

        data_file_name = os.path.join('..', 'data', parameters['data_file_name'])
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

    mesh = RectangleMesh(width, height)

    network_coordinates, bounding_box, network_edges, _, _, labels, _ \
        = data.read_graphml(data_file_name, with_labels=True)
    network_vertices = mesh.map_coordinates_to_support(
        np.array(network_coordinates), np.float64(0.8), bounding_box)
    nearest_vertex_indices = [mesh.nearest_vertex(network_vertex).index()
                              for network_vertex in network_vertices]
    network_convex_hulls = convex_hull.compute_connected_convex_hulls(
        network_vertices, network_edges)

    if use_postprocessing:
        # TODO: Make this a call into linear_geodesic_optimization.data
        vertices = mesh.get_coordinates()
        x = list(sorted(set(vertices[:,0])))
        y = list(sorted(set(vertices[:,1])))
        distances = np.array([
            convex_hull.distance_to_convex_hulls(
                np.array([px, py]),
                network_vertices,
                network_convex_hulls
            )
            for px in x
            for py in y
        ])
        z = z - np.array(z_0)
        z = (z - np.amin(z[distances == 0.], initial=np.amin(z))) \
            * np.exp(-1000 * distances**2)
        z = z - np.amin(z)
        if np.amax(z) != np.float64(0.):
            z = 0.15 * z / np.amax(z)
    mesh.set_parameters(z)

    geodesics = get_geodesics(mesh, network_coordinates, bounding_box,
                              network_edges, labels, True)

    with open(os.path.join(directory, 'geodesics.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['source', 'destination', 'geodesic_distance'])
        for (a, b), geodesic in geodesics:
            writer.writerow([a, b, geodesic])
