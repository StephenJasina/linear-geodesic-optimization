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
from linear_geodesic_optimization.optimization.curvature \
    import Computer as Curvature
from linear_geodesic_optimization.optimization.geodesic_loss \
    import Computer as GeodesicLoss
from linear_geodesic_optimization.optimization.laplacian \
    import Computer as Laplacian
from linear_geodesic_optimization.plot import get_line_plot, \
    get_scatter_plot, get_heat_map, get_mesh_plot


maxiters = 1000

def get_beta(x, y):
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    x_x = x @ x
    x_y = x @ y
    nu_0 = sum_y * x_x - x_y * sum_x
    nu_1 = n * x_y - sum_x * sum_y
    delta = n * x_x - sum_x * sum_x
    return (nu_0 / delta, nu_1 / delta)

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
        lambda_curvature = parameters['lambda_curvature']
        lambda_smooth = parameters['lambda_smooth']
        lambda_geodesic = parameters['lambda_geodesic']
        initial_radius = parameters['initial_radius']
        width = parameters['width']
        height = parameters['height']

    data_file_path = os.path.join('..', 'data', data_file_name)
    data_name, data_type = os.path.splitext(os.path.basename(data_file_name))

    mesh = RectangleMesh(width, height)

    coordinates, network_edges, network_curvatures, network_latencies = data.read_graphml(data_file_path)
    network_vertices = mesh.map_coordinates_to_support(np.array(coordinates))
    network_convex_hull = convex_hull.compute_convex_hull(network_vertices)

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
                beta_0, beta_1 = get_beta(estimated_latencies, true_latencies)
                before_data = (true_latencies, beta_0 + beta_1 * estimated_latencies)

        path_next = os.path.join(directory, str(i + 1))
        if i == maxiters or not os.path.exists(path_next):
            beta_0, beta_1 = get_beta(estimated_latencies, true_latencies)
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

    vertices = mesh.get_coordinates()
    x = list(sorted(set(vertices[:,0])))
    y = list(sorted(set(vertices[:,1])))
    z = vertices[:,2]

    # # Smooth using convex hull
    # distances = np.array([
    #     np.linalg.norm(np.array([px, py]) - convex_hull.project_to_convex_hull([px, py], network_vertices, network_convex_hull))
    #     for px in x
    #     for py in y
    # ])
    # z = z - np.array(z_0)
    # z = (z - np.amin(z)) * np.exp(-100 * distances**2)
    # z = z - np.amin(z)
    # z = 0.10 * z / np.amax(z)

    mesh.set_parameters(z)

    # Transpose z so that the heatmap is correct.
    z = z.reshape((width, height)).T

    figures['altitude'] = get_heat_map(x, y, z, 'Altitude' + lambda_string,
                                       network_vertices, network_edges, network_curvatures)

    laplacian = Laplacian(mesh)
    curvature = Curvature(mesh, laplacian)
    curvature.forward()
    kappa = np.array(curvature.kappa_G).reshape(width, height).T
    kappa[0,:] = 0.
    kappa[-1,:] = 0.
    kappa[:,0] = 0.
    kappa[:,-1] = 0.
    figures['curvature'] = get_heat_map(x, y, kappa, 'Curvature' + lambda_string,
                                        network_vertices, network_edges, network_curvatures)

    if sum(len(arr) for arr in before_data) > 0:
        figures['scatter'] = get_scatter_plot(before_data, after_data,
                                              'Latency Prediction' + lambda_string)

    # figures['mesh_plot'] = get_mesh_plot(mesh, 'Mesh' + lambda_string)

    # for filename, figure in figures.items():
    #     figure.savefig(os.path.join(directory, filename + '.png'), dpi=500)
    plt.show()
