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
    probes_filename, latencies_filenames,
    latency_threshold, clustering_distance, ricci_curvature_alpha,
    lambda_curvature, lambda_smooth,
    initial_radius, sides, mesh_scale,
    maxiter=None, output_dir_name=os.path.join('..', 'out'),
    initialization_file_path=None
):
    # Construct the mesh
    width = height = sides
    mesh = RectangleMesh(width, height, mesh_scale)

    # Construct the network graph
    probes_file_path = os.path.join('..', 'data', probes_filename)
    if isinstance(latencies_filenames, pathlib.PurePath):
        latencies_filenames = str(latencies_filenames)
    if isinstance(latencies_filenames, str):
        latencies_filenames = [latencies_filenames]
    latencies_file_paths = [
        os.path.join('..', 'data', latencies_filename)
        for latencies_filename in latencies_filenames
    ]
    graph = input_network.get_graph_from_paths(
        probes_file_path, latencies_file_paths[0],
        latency_threshold, clustering_distance,
        ricci_curvature_alpha=ricci_curvature_alpha
    )
    network = input_network.get_network_data(graph)
    graph_data, vertex_data, edge_data = network
    bounding_box = graph_data['bounding_box']
    network_coordinates = graph_data['coordinates']
    network_vertices = mesh.map_coordinates_to_support(np.array(network_coordinates), np.float64(0.8), bounding_box)
    network_edges = graph_data['edges']
    network_curvatures = edge_data['ricciCurvature']
    for latencies_file_path, network_weight in zip(latencies_file_paths[1:], network_weights[1:]):
        _, _, network_edges_to_add, network_curvatures_to_add, _ = \
            input_network.extract_from_graph_old(
                input_network.get_graph_from_paths(
                    probes_file_path, latencies_file_path,
                    latency_threshold, clustering_distance,
                    ricci_curvature_alpha=ricci_curvature_alpha
                )
            )
        network_edges.append(network_edges_to_add)
        network_curvatures.append(network_curvatures_to_add)

    # Setup snapshots
    directory = os.path.join(
        output_dir_name,
        f'{lambda_curvature}_{lambda_smooth}_{initial_radius}_{width}_{height}_{mesh_scale}'
    )
    if os.path.isdir(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

    parameters = {
        'probes_filename': probes_filename,
        'latencies_filenames':latencies_filenames,
        'epsilon': latency_threshold,
        'clustering_distance': clustering_distance,
        'should_remove_TIVs': False, # TODO: Pass this as a parameter?
        'ricci_curvature_alpha': ricci_curvature_alpha,
        'lambda_curvature': lambda_curvature,
        'lambda_smooth': lambda_smooth,
        'network_weights': network_weights,
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
        lambda_curvature, lambda_smooth, network_weights,
        directory
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
    probes_filenames = [
        pathlib.PurePath('..', 'data', 'toy', 'toy_probes.csv'),
        pathlib.PurePath('..', 'data', 'toy', 'elbow_probes.csv'),
    ]

    latencies_filenames = [
        pathlib.PurePath('..', 'data', 'toy', 'toy_latencies.csv'),
        pathlib.PurePath('..', 'data', 'toy', 'elbow_latencies.csv'),
    ]

    output_dir_names = [
        pathlib.PurePath('..', 'outputs', 'toy', 'toy'),
        pathlib.PurePath('..', 'outputs', 'toy', 'elbow'),
    ]

    count = len(probes_filenames)

    latency_thresholds = [0] * count
    clustering_distances = [None] * count

    lambda_curvatures = [1.] * count
    lambda_smooths = [0.002] * count
    ricci_curvature_alphas = [0.9999] * count
    initial_radii = [20.] * count
    sides = [50] * count
    mesh_scales = [1.] * count

    max_iters = [2000] * count

    arguments = list(zip(
        probes_filenames, latencies_filenames,
        latency_thresholds, clustering_distances, ricci_curvature_alphas,
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
