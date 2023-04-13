import itertools
import json
import os
import pickle
import sys

from matplotlib import pyplot as plt
import numpy as np

from linear_geodesic_optimization import convex_hull
from linear_geodesic_optimization import data
from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh
from linear_geodesic_optimization.optimization import curvature, linear_regression
from linear_geodesic_optimization.plot import get_line_plot, \
    get_scatter_plot, get_heat_map, get_mesh_plot

maxiters = 250

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

        # TODO: Simplify this logic
        if 'data_file_name' in parameters:
            data_file_name = parameters['data_file_name']
        else:
            data_name = parameters['data_name']
            data_file_name = data_name + '.json'
        lambda_geodesic = parameters['lambda_geodesic']
        lambda_smooth = parameters['lambda_smooth']
        lambda_curvature = parameters['lambda_curvature']
        initial_radius = parameters['initial_radius']
        width = parameters['width']
        height = parameters['height']

    data_file_path = os.path.join('..', 'data', data_file_name)
    data_name, data_type = os.path.splitext(os.path.basename(data_file_name))

    mesh = RectangleMesh(width, height)

    if data_type == '.json':
        coordinates, network_edges, network_curvatures, network_latencies = data.read_json(data_file_path)
    elif data_type == '.graphml':
        coordinates, network_edges, network_curvatures, network_latencies = data.read_graphml(data_file_path)
    network_vertices = mesh.map_coordinates_to_support(coordinates)
    network_convex_hull = convex_hull.compute_convex_hull(network_vertices)
    latencies = data.map_latencies_to_mesh(mesh, network_vertices, network_latencies)

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
                z_0 = data['mesh_parameters']
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

    vertices = mesh.get_vertices()
    x = list(sorted(set(vertices[:,0])))
    y = list(sorted(set(vertices[:,1])))
    z = vertices[:,2] - 1. * np.array(z_0)

    # Smooth using convex hull
    distances = np.array([
        np.linalg.norm(np.array([px, py]) - convex_hull.project_to_convex_hull([px, py], network_vertices, network_convex_hull))
        for py in y
        for px in x
    ])
    z = (z - np.amin(z)) * np.exp(-100 * distances**2)
    z = z - np.amin(z)

    # TODO: Remove this. This is here since the US data was initially
    # transposed on accident.
    z = mesh.set_parameters((z.reshape((width, height)).T).reshape((-1,)))

    mesh.set_parameters(z)

    z = z.reshape((width, height))

    figures['altitude'] = get_heat_map(x, y, z, 'Altitude' + lambda_string,
                                       network_vertices, network_edges, network_curvatures,
                                       [network_vertices[i] for i in network_convex_hull])

    curvature_forward = curvature.Forward(mesh)
    curvature_forward.calc()
    kappa = curvature_forward.kappa_G.reshape(width, height)[1:width - 1,1:height - 1]
    figures['curvature'] = get_heat_map(x, y, kappa, 'Curvature' + lambda_string,
                                        network_vertices, network_edges, network_curvatures, v_range=(-2., 5.))

    if sum(len(arr) for arr in before_data) > 0:
        figures['scatter'] = get_scatter_plot(before_data, after_data,
                                              'Latency Prediction' + lambda_string)

    figures['mesh_plot'] = get_mesh_plot(mesh, 'Mesh' + lambda_string)

    for filename, figure in figures.items():
        figure.savefig(os.path.join(directory, filename + '.png'), dpi=500)
    # plt.show()
