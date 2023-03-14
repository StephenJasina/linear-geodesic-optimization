import itertools
import json
import os
import pickle
import sys

from matplotlib import pyplot as plt
import numpy as np

from linear_geodesic_optimization.mesh.adaptive import Mesh as AdaptiveMesh
from linear_geodesic_optimization.optimization import curvature, linear_regression
from linear_geodesic_optimization.plot import get_line_plot, \
    get_scatter_plot, get_heat_map, get_mesh_plot

maxiters = 1000

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

        data_name = parameters['data_name']
        lambda_geodesic = parameters['lambda_geodesic']
        lambda_smooth = parameters['lambda_smooth']
        lambda_curvature = parameters['lambda_curvature']
        width = parameters['width']
        height = parameters['height']

    data_directory = os.path.join('..', 'data', data_name)

    # Get the network vertices
    coordinates = None
    label_to_index = {}
    with open(os.path.join(data_directory, 'position.json')) as f:
        position_json = json.load(f)

        label_to_index = {label: index for index, label in enumerate(position_json)}

        coordinates = [None for _ in range(len(position_json))]
        for vertex, position in position_json.items():
            coordinates[label_to_index[vertex]] = position

    network_vertices = AdaptiveMesh.map_coordinates_to_support(coordinates)

    # Get the network edges and curvatures
    network_edges = []
    network_curvatures = []
    with open(os.path.join(data_directory, 'curvature.json')) as f:
        curvature_json = json.load(f)

        for edge, network_curvature in curvature_json.items():
            u = label_to_index[edge[0]]
            v = label_to_index[edge[1]]

            network_edges.append((u, v))
            network_curvatures.append(network_curvature)

    # Create the mesh
    density = np.amin([
        np.linalg.norm(network_vertices[i] - network_vertices[j])
        for i, j in network_edges
    ]) / 10
    points = list(np.array(network_vertices))
    for i, j in network_edges:
        count = int(np.ceil(np.linalg.norm(network_vertices[i] - network_vertices[j]) / density))
        for p in np.linspace(network_vertices[i], network_vertices[j], count)[1:-1]:
            # Guard for floating point error
            if np.amin([np.linalg.norm(p - point) for point in points]) > 1e-10:
                points.append(p)
    mesh = AdaptiveMesh(width, height, points)

    L_geodesics = []
    L_smooths = []
    L_curvatures = []
    Ls = []

    before_data = None
    after_data = None
    linear_regression_forward = linear_regression.Forward()

    for i in itertools.count():
        path = os.path.join(directory, str(i))
        with open(path, 'rb') as f:
            data = pickle.load(f)

            mesh.set_parameters(data['mesh_parameters'])

            L_geodesic = data['L_geodesic']
            L_geodesics.append(L_geodesic)

            L_smooth = data['L_smooth']
            L_smooths.append(L_smooth)

            L_curvature = data['L_curvature']
            L_curvatures.append(L_curvature)

            Ls.append(lambda_geodesic * L_geodesic + lambda_smooth * L_smooth
                      + lambda_curvature * L_curvature)

            true_latencies = np.array(data['true_latencies'])
            estimated_latencies = np.array(data['estimated_latencies'])
            if i == 0:
                beta_0, beta_1 = linear_regression_forward.get_beta(estimated_latencies, true_latencies)
                before_data = (true_latencies, beta_0 + beta_1 * estimated_latencies)

        path_next = os.path.join(directory, str(i + 1))
        if i == maxiters or not os.path.exists(path_next):
            beta_0, beta_1 = linear_regression_forward.get_beta(estimated_latencies, true_latencies)
            after_data = (true_latencies, beta_0 + beta_1 * estimated_latencies)
            break

    figures = {}

    lambda_string = ' ($\lambda_{\mathrm{geodesic}} = ' + str(lambda_geodesic) \
        + '$, $\lambda_{\mathrm{curvature}} = ' + str(lambda_curvature) \
        + '$, $\lambda_{\mathrm{smooth}} = ' + str(lambda_smooth) + '$)'

    figures['geodesic_loss'] = get_line_plot(L_geodesics, 'Geodesic Loss' + lambda_string, maxiters)
    figures['smoothness_loss'] = get_line_plot(L_smooths, 'Smoothness Loss' + lambda_string, maxiters)
    figures['curvature_loss'] = get_line_plot(L_curvatures, 'Curvature Loss' + lambda_string, maxiters, 2.25)
    figures['total_loss'] = get_line_plot(Ls, 'Total Loss' + lambda_string, maxiters, 2.25)

    figures['scatter'] = get_scatter_plot(before_data, after_data,
                                          'Latency Prediction' + lambda_string)

    # TODO: Figure out heatmaps

    figures['mesh_plot'] = get_mesh_plot(mesh, 'Mesh' + lambda_string)

    for filename, figure in figures.items():
        figure.savefig(os.path.join(directory, filename + '.png'), dpi=500)
    # plt.show()
