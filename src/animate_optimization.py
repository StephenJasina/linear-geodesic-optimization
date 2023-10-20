import itertools
import json
import os
import pickle
import sys

from matplotlib import pyplot as plt
from matplotlib import animation as animation
import numpy as np

from linear_geodesic_optimization import convex_hull
from linear_geodesic_optimization import data
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

        data_file_name = parameters['data_file_name']
        width = parameters['width']
        height = parameters['height']
        scale = parameters['scale']

    data_file_path = os.path.join('..', 'data', data_file_name)
    data_name, _ = os.path.splitext(os.path.basename(data_file_name))

    mesh = RectangleMesh(width, height, scale)
    vertices = mesh.get_coordinates()
    x = list(sorted(set(vertices[:,0])))
    y = list(sorted(set(vertices[:,1])))

    network_coordinates, bounding_box, network_edges, _, _ \
        = data.read_graphml(data_file_path, None)
    network_vertices = mesh.map_coordinates_to_support(
        np.array(network_coordinates), np.float64(0.8), bounding_box)
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
