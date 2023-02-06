import itertools
import json
import os
import pickle
import sys

from matplotlib import pyplot as plt
import numpy as np

from linear_geodesic_optimization.optimization import curvature
from linear_geodesic_optimization.plot import get_line_plot, \
    get_scatter_plot, get_heat_map

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python3 <directory name>')
        sys.exit(0)

    directory = sys.argv[1]

    if not os.path.exists(os.path.join(directory, '0')):
        print('Error: supplied directory must contain file named "0"')
        sys.exit(0)

    # TODO: Automate this (save it when doing the computations)
    toy_directory = os.path.join('..', 'data', 'two_islands')

    mesh = None

    L_geodesics = []
    L_smooths = []
    L_curvatures = []
    Ls = []

    before_data = None
    after_data = None

    for i in itertools.count():
        path = os.path.join(directory, str(i))
        with open(path, 'rb') as f:
            data = pickle.load(f)

            mesh = data['mesh']

            L_geodesic = data['L_geodesic']
            L_geodesics.append(L_geodesic)

            L_smooth = data['L_smooth']
            L_smooths.append(L_smooth)

            L_curvature = data['L_curvature']
            L_curvatures.append(L_curvature)

            # TODO: Only save these once, probably
            lambda_geodesic = data['lambda_geodesic']
            lambda_smooth = data['lambda_smooth']
            lambda_curvature = data['lambda_curvature']
            Ls.append(lambda_geodesic * L_geodesic + lambda_smooth * L_smooth
                      + lambda_curvature * L_curvature)

            true_latencies = data['true_latencies']
            estimated_latencies = data['estimated_latencies']
            if i == 0:
                print('Using')
                print(f'\tlambda_geodesic = {lambda_geodesic}')
                print(f'\tlambda_smooth = {lambda_smooth}')
                print(f'\tlambda_curvature = {lambda_curvature}')
                before_data = (true_latencies, estimated_latencies)

            path_next = os.path.join(directory, str(i + 1))
            if not os.path.exists(path_next):
                after_data = (true_latencies, estimated_latencies)
                break

    figures = []

    lambda_string = ' ($\lambda_{\mathrm{smooth}} = ' + str(lambda_smooth) \
        + '$, $\lambda_{\mathrm{curvature}} = ' + str(lambda_curvature) + '$)'

    figures.append(get_line_plot(L_geodesics, 'Geodesic Loss' + lambda_string))
    figures[-1].savefig(os.path.join(directory, 'geodesic_loss.png'))

    figures.append(get_line_plot(L_smooths, 'Smoothness Loss' + lambda_string))
    figures[-1].savefig(os.path.join(directory, 'smoothness_loss.png'))

    figures.append(get_line_plot(L_curvatures,
                                 'Curvature Loss' + lambda_string))
    figures[-1].savefig(os.path.join(directory, 'curvature_loss.png'))

    figures.append(get_line_plot(Ls, 'Total Loss' + lambda_string))
    figures[-1].savefig(os.path.join(directory, 'total_loss.png'))

    vertices = mesh.get_vertices()
    coordinates = None
    label_to_index = {}
    with open(os.path.join(toy_directory, 'position.json')) as f:
        position_json = json.load(f)

        label_to_index = {label: index
                          for index, label in enumerate(position_json)}

        coordinates = [None for _ in range(len(position_json))]
        for vertex, position in position_json.items():
            coordinates[label_to_index[vertex]] = position

    network_vertices = mesh.map_coordinates_to_support(coordinates)

    network_edges = []
    with open(os.path.join(toy_directory, 'latency.json')) as f:
        latency_json = json.load(f)

        for edge, latency in latency_json.items():
            u = label_to_index[edge[0]]
            v = label_to_index[edge[1]]

            network_edges.append((u, v))

    x = list(sorted(set(vertices[:,0])))
    y = list(sorted(set(vertices[:,1])))
    z = vertices[:,2].reshape(len(x), len(y), order='F')

    width = len(x)
    height = len(y)

    x = x[1:width - 1]
    y = y[1:height - 1]
    z = z[1:width - 1,1:height - 1]
    z = z - np.amin(z)

    figures.append(get_heat_map(x, y, z, 'Altitude' + lambda_string,
                                network_vertices, network_edges))
    figures[-1].savefig(os.path.join(directory, 'altitude.png'))

    curvature_forward = curvature.Forward(mesh, [], [], [], 0.)
    curvature_forward.calc()
    kappa = curvature_forward.kappa.reshape(width, height, order='F')[1:width - 1,1:height - 1]
    figures.append(get_heat_map(x, y, kappa, 'Curvature' + lambda_string,
                   network_vertices, network_edges))
    figures[-1].savefig(os.path.join(directory, 'curvature.png'))

    figures.append(get_scatter_plot(before_data, after_data,
                                    'Latency Prediction' + lambda_string))
    figures[-1].savefig(os.path.join(directory, 'scatter.png'))

    for figure in figures:
        figure.show()
    plt.show()
