import csv
import os
import pickle
import sys

from matplotlib import pyplot as plt
import numpy as np

from linear_geodesic_optimization import convex_hull
from linear_geodesic_optimization import data
from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh
from linear_geodesic_optimization.mesh.sphere import Mesh as SphereMesh
from linear_geodesic_optimization.optimization import geodesic, linear_regression
from linear_geodesic_optimization import plot

max_iterations = 250

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

    coordinates, _, _, _, labels = data.read_graphml(data_file_name, with_labels=True)
    network_vertices = mesh.map_coordinates_to_support(coordinates)
    nearest_vertex_indices = [mesh.nearest_vertex_index(network_vertex) for network_vertex in network_vertices]
    network_convex_hull = convex_hull.compute_convex_hull(network_vertices)

    vertices = mesh.get_vertices()
    x = list(sorted(set(vertices[:,0])))
    y = list(sorted(set(vertices[:,1])))
    z = z - z_0
    distances = np.array([
        np.linalg.norm(np.array([px, py]) - convex_hull.project_to_convex_hull([px, py], network_vertices, network_convex_hull))
        for py in y
        for px in x
    ])
    z = (z - np.amin(z)) * np.exp(-100 * distances**2)
    z = z - np.amin(z)
    mesh.set_parameters(z)

    geodesic_forward = geodesic.Forward(mesh)
    geodesics = {}
    for mesh_i, label_i in zip(nearest_vertex_indices, labels):
        geodesic_forward.calc(mesh_i)
        for mesh_j, label_j in zip(nearest_vertex_indices, labels):
            geodesics[label_i,label_j] = geodesic_forward.phi[mesh_j]

    with open(os.path.join(directory, 'geodesics.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['source', 'destination', 'geodesic_distance'])
        for (a, b), geodesic in geodesics.items():
            writer.writerow([a, b, geodesic])

