import os
import pickle

import networkx as nx
import pandas as pd
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
from matplotlib import animation as animation
from mpl_toolkits.basemap import Basemap
import numpy as np

from linear_geodesic_optimization import data
from linear_geodesic_optimization.plot import get_rectangular_mesh_plot
from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh

directory = os.path.join('..', 'out_US_fine_threshold_nonsequential_smooth')

lambda_curvature = 1.
lambda_smooth = 0.004
lambda_geodesic = 0.
initial_radius = 20.
width = 50
height = 50
scale = 1.
subdirectory_name = f'{lambda_curvature}_{lambda_smooth}_{lambda_geodesic}_{initial_radius}_{width}_{height}_{scale}'

fps = 24
# length in seconds
animation_length = 10

def get_image_data(data_file_path, resolution=100):
    coordinates, _, _, _, _ = data.read_graphml(data_file_path)
    coordinates = np.array(coordinates)
    center = np.mean(coordinates, axis=0)
    scale_factor = 0.8
    coordinates = center + (coordinates - center) / scale_factor

    coordinates_left = np.amin(coordinates[:,0])
    coordinates_right = np.amax(coordinates[:,0])
    coordinates_bottom = np.amin(coordinates[:,1])
    coordinates_top = np.amax(coordinates[:,1])

    if coordinates_right - coordinates_left > coordinates_top - coordinates_bottom:
        center = (coordinates_bottom + coordinates_top) / 2.
        scale_factor = (coordinates_top - coordinates_bottom) / (coordinates_right - coordinates_left)
        coordinates_bottom = center + (coordinates_bottom - center) / scale_factor
        coordinates_top = center + (coordinates_top - center) / scale_factor
    else:
        center = (coordinates_left + coordinates_right) / 2.
        scale_factor = (coordinates_right - coordinates_left) / (coordinates_top - coordinates_bottom)
        coordinates_left = center + (coordinates_left - center) / scale_factor
        coordinates_right = center + (coordinates_right - center) / scale_factor

    left, _ = data.inverse_mercator(coordinates_left, 0.)
    right, _ = data.inverse_mercator(coordinates_right, 0.)
    _, bottom = data.inverse_mercator(0., coordinates_bottom)
    _, top = data.inverse_mercator(0., coordinates_top)

    map = Basemap(llcrnrlon=left, urcrnrlon=right,
                  llcrnrlat=bottom, urcrnrlat=top, epsg=3857)
    fig, ax = plt.subplots()
    image_data = map.arcgisimage(service='USA_Topo_Maps', ax=ax,
                                 xpixels=resolution, y_pixels=resolution).get_array()
    image_data = np.flipud(image_data).swapaxes(0, 1) / 255
    plt.close(fig)
    return image_data

if __name__ == '__main__':
    epsilons = []
    entry_prefix = 'graph_'
    with os.scandir(directory) as it:
        for entry in it:
            if entry.name.startswith(entry_prefix) and not entry.is_file():
                epsilons.append((float(entry.name[len(entry_prefix):]), entry.name))
    epsilons = list(sorted(epsilons))
    manifold_count = len(epsilons)

    initialization_path = os.path.join(directory, epsilons[0][1], subdirectory_name, '0')

    zs = {}
    for epsilon, entry_name in epsilons:
        print(f'Reading data from cutoff {epsilon}')

        current_directory = os.path.join(
            directory, entry_name, subdirectory_name
        )

        iteration = max(
            int(name)
            for name in os.listdir(current_directory)
            if name.isdigit()
        )
        path = os.path.join(current_directory, str(iteration))
        mesh = data.get_mesh_output(
            current_directory, postprocessed=True,
            initialization_path=initialization_path
        )

        zs[epsilon] = mesh.get_parameters()

    z_max = np.amax(list(zs.values()))

    mesh = RectangleMesh(width, height, scale)

    with open(os.path.join(directory, epsilons[0][1], subdirectory_name, 'parameters'), 'rb') as f:
        parameters = pickle.load(f)
        data_file_name = parameters['data_file_name']
        data_file_path = os.path.join('..', 'data', data_file_name)
    resolution = 500
    face_colors = get_image_data(data_file_path, resolution)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_facecolor((0.5, 0.5, 0.5))

    def get_frame(epsilon):
        print(f'Computing frame for cutoff {epsilon}')

        for left_index in range(len(epsilons) + 1):
            if left_index == len(epsilons) or epsilons[left_index][0] > epsilon:
                break
        left_index -= 1
        for right_index in range(len(epsilons) - 1, -2, -1):
            if right_index == -1 or epsilons[right_index][0] < epsilon:
                break
        right_index += 1

        left = epsilons[left_index][0]
        right = epsilons[right_index][0]

        lam = 0. if right == left else (epsilon - left) / (right - left)
        z = (1 - lam) * zs[left] + lam * zs[right]
        z = np.reshape(z, (width, height))

        print(left, right, lam)

        entry_name = epsilons[left_index if lam < 0.5 else right_index][1]
        with open(os.path.join(directory, entry_name, subdirectory_name, 'parameters'), 'rb') as f:
            parameters = pickle.load(f)
            data_file_name = parameters['data_file_name']
            data_file_path = os.path.join('..', 'data', data_file_name)
            coordinates, bounding_box, network_edges, network_curvatures, \
            network_latencies, network_nodes, network_city \
                = data.read_graphml(data_file_path, with_labels=True)
        coordinates = np.array(coordinates)
        network_vertices = mesh.map_coordinates_to_support(coordinates, np.float64(0.8), bounding_box)

        ax.clear()

        return [
            get_rectangular_mesh_plot(z, face_colors, None,
                          np.amax(z) / z_max * 0.25,
                          [network_vertices, network_edges, network_curvatures, network_city],
                          ax),
            ax.text2D(0.05, 0.95, f'{epsilon}',
                      transform=ax.transAxes),
        ]


    ani = animation.FuncAnimation(fig, get_frame,
                                  np.linspace(epsilons[0][0], epsilons[-1][0],
                                              int(animation_length * fps + 1)),
                                  interval=1000/fps,
                                  blit=True)
    ani.save(os.path.join('..', 'animation_US_nonsequential_smooth.mp4'), dpi=300)
