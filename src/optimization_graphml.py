import concurrent.futures
import itertools
import json
import os
import pathlib
# TODO: Convert to plain text
import pickle
import shutil
import time
import warnings

import networkx as nx
import numpy as np
import scipy

from linear_geodesic_optimization.data import input_network
from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh
from linear_geodesic_optimization.optimization import optimization


# Error on things like division by 0
warnings.simplefilter('error')

def main(
    graphml_filename,
    lambda_curvature, lambda_smooth,
    initial_radius=20., sides=50, mesh_scale=1.,
    maxiter=None, output_dir_name=os.path.join('..', 'out'),
    initialization_file_path=None
):
    # Construct the mesh
    width = height = sides
    mesh = RectangleMesh(width, height, mesh_scale)

    # Construct the network graph
    network = input_network.extract_from_graph(nx.read_graphml(graphml_filename))
    network_coordinates, bounding_box, network_edges, network_curvatures, _ = network
    network_vertices = mesh.map_coordinates_to_support(np.array(network_coordinates), np.float64(0.8), bounding_box)
    network_edges = [network_edges]
    network_curvatures = [network_curvatures]

    # Setup snapshots
    directory = os.path.join(
        output_dir_name,
        f'{lambda_curvature}_{lambda_smooth}_{initial_radius}_{width}_{height}_{mesh_scale}'
    )
    if os.path.isdir(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

    parameters = {
        'graphml_filename': graphml_filename,
        'should_remove_TIVs': False, # TODO: Pass this as a parameter?
        'lambda_curvature': lambda_curvature,
        'lambda_smooth': lambda_smooth,
        'initial_radius': initial_radius,
        'width': width,
        'height': height,
        'mesh_scale': mesh_scale,
        'coordinates_scale': 0.8
    }

    with open(os.path.join(directory, 'parameters'), 'wb') as f:
        pickle.dump(parameters, f)

    # Initialize mesh
    if initialization_file_path is None:
        z_0 = np.array([
            (initial_radius**2
                - (i / (width - 1) - 0.5)**2
                - (j / (height - 1) - 0.5)**2)**0.5
            for i in range(width)
            for j in range(height)
        ]).reshape((width * height,))
        z_0 = z_0 - np.amin(z_0)
    else:
        with open(initialization_file_path, 'rb') as f:
            z_0 = np.array(pickle.load(f)['mesh_parameters'])
    z_0 = mesh.set_parameters(z_0)

    computer = optimization.Computer(
        mesh, network_vertices, network_edges, network_curvatures,
        1.01 * 2**0.5 * mesh_scale / width,
        lambda_curvature, lambda_smooth,
        directory=directory
    )

    f = computer.forward
    g = computer.reverse

    computer.diagnostics(None)
    minimizer_kwargs = {
        'method': 'L-BFGS-B',
        'jac': g,
        'callback': computer.diagnostics,
        'options': None if maxiter is None else {'maxiter': maxiter},
    }
    z = scipy.optimize.minimize(f, z_0, **minimizer_kwargs).x
    with open(os.path.join(directory, 'output'), 'wb') as f:
        pickle.dump({
            'parameters': parameters,
            'initial': optimization.Computer.to_float_list(z_0),
            'final': optimization.Computer.to_float_list(z),
            'network': network, # TODO: JSONify this
        }, f)


if __name__ == '__main__':
    graphml_directory = pathlib.PurePath('..', 'data', 'Internet2', 'graphml')
    graphml_filenames = list(sorted(
        graphml_directory / filename
        for filename in os.listdir(graphml_directory)
    ))[:25]

    count = len(graphml_filenames)

    lambda_curvatures = [1.] * count
    lambda_smooths = [0.005] * count
    initial_radii = [20.] * count
    sides = [50] * count
    mesh_scales = [1.] * count

    max_iters = [1] * count

    output_dir_names = [
        pathlib.PurePath('..', 'outputs', 'Internet2', 'fake_animation', graphml_filename.name)
        for graphml_filename in graphml_filenames
    ]

    arguments = list(zip(
        graphml_filenames,
        lambda_curvatures, lambda_smooths,
        initial_radii, sides, mesh_scales,
        max_iters,
        output_dir_names,
    ))
    # Need to use ProcessPoolExecutor instead of multiprocessing.Pool
    # to allow child processes to spawn their own subprocesses
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for _ in executor.map(main, *zip(*arguments)):
            pass
