import itertools
import json
import os
import pickle
import sys

from matplotlib import pyplot as plt
from matplotlib import animation as animation
import numpy as np

sys.path.append('.')
from linear_geodesic_optimization import convex_hull
from linear_geodesic_optimization.data import input_network
from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh
from linear_geodesic_optimization.plot import get_mesh_plot


maxiters = 10000
frame_step = 50

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python3 <directory name>')
        sys.exit(0)

    directory = sys.argv[1]

    if not os.path.exists(os.path.join(directory, '0')):
        print('Error: supplied directory must contain file named "0"')
        sys.exit(0)

    if not os.path.exists(os.path.join(directory, 'parameters')):
        print('Error: supplied directory must contain file named "parameters"')
        sys.exit(0)

    with open(os.path.join(directory, 'parameters'), 'rb') as f:
        parameters = pickle.load(f)

        probes_filename = parameters['probes_filename']
        latencies_filename = parameters['latencies_filename']
        epsilon = parameters['epsilon']
        clustering_distance = parameters['clustering_distance']
        should_remove_tivs = parameters['should_remove_TIVs']
        width = parameters['width']
        height = parameters['height']
        mesh_scale = parameters['mesh_scale']
        coordinates_scale = parameters['coordinates_scale']

    probes_file_path = os.path.join('..', 'data', probes_filename)
    latencies_file_path = os.path.join('..', 'data', latencies_filename)

    mesh = RectangleMesh(width, height, mesh_scale)
    vertices = mesh.get_coordinates()
    x = list(sorted(set(vertices[:,0])))
    y = list(sorted(set(vertices[:,1])))

    network, latencies = input_network.get_graph(
        probes_file_path, latencies_file_path,
        epsilon, clustering_distance, should_remove_tivs,
        should_include_latencies=True
    )
    network_coordinates, bounding_box, network_edges, _, _ \
        = input_network.extract_from_graph(network, latencies)
    network_vertices = mesh.map_coordinates_to_support(
        np.array(network_coordinates), coordinates_scale, bounding_box)
    network_convex_hulls = convex_hull.compute_connected_convex_hulls(
        network_vertices, network_edges)

    artists = []
    fig = None
    ax = None

    frame_step = max(frame_step, 1)
    for i in itertools.count(0, frame_step):
        print(i)
        path = os.path.join(directory, str(i))
        with open(path, 'rb') as f:
            data = pickle.load(f)

            z = data['mesh_parameters']

            if i == 0:
                z_0 = z

            # Smooth using convex hull
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

            if i == 0:
                fig = get_mesh_plot(mesh, 'Mesh', False)
                ax = fig.gca()
            artist = get_mesh_plot(mesh, 'Mesh', False, ax)
            artists.append([artist])

        path_next = os.path.join(directory, str(i + frame_step))
        if i == maxiters or not os.path.exists(path_next):
            break

    ani = animation.ArtistAnimation(fig, artists,
                                    interval = 20000 / len(artists))
    ani.save("movie.mp4")
