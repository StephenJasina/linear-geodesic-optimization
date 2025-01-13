import concurrent.futures
import itertools
import json
import os
import pathlib
import shutil
import time
import typing
import warnings

import networkx as nx
import numpy as np
import scipy

from linear_geodesic_optimization.data import input_network
from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh
from linear_geodesic_optimization.optimization import optimization


# Error on things like division by 0
warnings.simplefilter('error')

directory_data = pathlib.PurePath('..', 'data')
directory_outputs = pathlib.PurePath('..', 'outputs', 'geodesics')

# TODO: Do we want something like this?
def rectangularize_parameters(mesh: RectangleMesh) -> typing.List[typing.List[float]]:
    width = mesh.get_width()
    height = mesh.get_height()

    parameters = [
        [
            None
            for _ in range(height)
        ]
        for _ in range(width)
    ]
    for index, parameter in zip(mesh.get_trim_mapping(), mesh.get_parameters()):
        parameters[index // height][index % height] = float(parameter)

    return parameters

def main(
    filename_probes, filename_links,
    latency_threshold, clustering_distance, ricci_curvature_alpha,
    lambda_curvature, lambda_smooth,
    initial_radius, sides, mesh_scale, coordinates_scale, network_trim_radius,
    directory_output, maxiter=None,
    initialization_file_path=None
):
    # Construct the mesh
    width = height = sides
    mesh = RectangleMesh(width, height, mesh_scale)

    # Construct the network graph
    file_path_probes = directory_data / filename_probes
    file_path_links = directory_data / filename_links
    graph = input_network.get_graph_from_paths(
        file_path_probes, file_path_links,
        epsilon=latency_threshold,
        clustering_distance=clustering_distance,
        ricci_curvature_alpha=ricci_curvature_alpha,
        ricci_curvature_weight_label='throughput'
    )
    network = input_network.get_network_data(graph)
    graph_data, vertex_data, edge_data = network
    bounding_box = graph_data['bounding_box']
    network_coordinates = graph_data['coordinates']
    network_vertices = mesh.map_coordinates_to_support(np.array(network_coordinates), coordinates_scale, bounding_box)
    network_edges = graph_data['edges']
    network_curvatures = edge_data['ricciCurvature']

    # Setup snapshots
    directory_output = directory_output / f'{lambda_curvature}_{lambda_smooth}_{initial_radius}_{width}_{height}_{mesh_scale}'
    if os.path.isdir(directory_output):
        shutil.rmtree(directory_output)
    os.makedirs(directory_output)

    parameters = {
        'probes_filename': str(filename_probes),
        'latencies_filenames':str(filename_links),
        'epsilon': float(latency_threshold),
        'clustering_distance': float(clustering_distance) if clustering_distance is not None else None,
        'should_remove_TIVs': False, # TODO: Pass this as a parameter?
        'ricci_curvature_alpha': float(ricci_curvature_alpha),
        'lambda_curvature': float(lambda_curvature),
        'lambda_smooth': float(lambda_smooth),
        'initial_radius': float(initial_radius),
        'width': int(width),
        'height': int(height),
        'mesh_scale': float(mesh_scale),
        'coordinates_scale': float(coordinates_scale),
        'network_trim_radius': float(network_trim_radius),
    }

    with open(directory_output / 'parameters.json', 'w') as file_parameters:
        json.dump(
            parameters, file_parameters,
            ensure_ascii=False, indent=4, sort_keys=True
        )

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
        with open(initialization_file_path, 'r') as f:
            z_0 = np.array(json.load(f)['final'])
    z_0 = mesh.set_parameters(z_0)

    if not np.isposinf(network_trim_radius):
        mesh.trim_to_graph(network_vertices, network_edges, network_trim_radius)
        z_0 = mesh.get_parameters()

    computer = optimization.Computer(
        mesh, network_vertices, network_edges, network_curvatures,
        1.01 * 2**0.5 * mesh_scale / width,
        lambda_curvature, lambda_smooth,
        directory=directory_output
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

    z = mesh.set_parameters(z)

    with open(directory_output / 'output.json', 'w') as file_output:
        json.dump({
            'parameters': parameters,
            'initial': z_0.tolist(),
            'final': z.tolist(),
            'network': network,
        }, file_output, ensure_ascii=False)

if __name__ == '__main__':
    directory_links = pathlib.PurePath('toy', 'two_clusters', 'throughputs')
    filenames_links = list(sorted(
        directory_links / filename
        for filename in os.listdir(directory_data / directory_links)
    ))[:1]

    count = len(filenames_links)

    filenames_probes = [
        pathlib.PurePath('toy', 'two_clusters', 'probes.csv'),
    ] * count

    directories_output = [
        directory_outputs / pathlib.PurePath('toy', 'two_clusters', filename_links.stem)
        for filename_links in filenames_links
    ]

    latency_thresholds = [0] * count
    clustering_distances = [None] * count

    lambdas_curvature = [1.] * count
    lambdas_smooth = [0.005] * count
    ricci_curvature_alphas = [0.9999] * count
    initial_radii = [20.] * count
    sides = [50] * count
    mesh_scales = np.array([0.7] * count)
    coordinates_scales = np.array([0.8] * count)

    network_trim_radii = 0.2 * mesh_scales

    max_iters = [2000] * count

    arguments = list(zip(
        filenames_probes, filenames_links,
        latency_thresholds, clustering_distances, ricci_curvature_alphas,
        lambdas_curvature, lambdas_smooth,
        initial_radii, sides, mesh_scales, coordinates_scales, network_trim_radii,
        directories_output, max_iters,
    ))
    # Need to use ProcessPoolExecutor instead of multiprocessing.Pool
    # to allow child processes to spawn their own subprocesses
    with concurrent.futures.ProcessPoolExecutor(9) as executor:
        for _ in executor.map(main, *zip(*arguments)):
            pass
