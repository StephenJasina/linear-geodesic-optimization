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

def main(
    *,  # All parameters are keyword only
    filename_probes = None,
    filename_links = None,
    filename_graphml = None,
    latency_threshold = None,
    clustering_distance = None,
    ricci_curvature_alpha = 0.9999,
    lambda_curvature = 1.,
    lambda_smooth = 0.,
    initial_radius,
    sides,
    mesh_scale = 1.,
    coordinates_scale = 0.8,
    network_trim_radius = None,
    directory_output,
    maxiter = None,
    initialization_file_path = None
):
    # Construct the mesh
    width = height = sides
    mesh = RectangleMesh(width, height, mesh_scale)

    # Construct the networkx graph
    if filename_graphml is not None:
        graph = nx.read_graphml(directory_data / filename_graphml)
    elif filename_probes is not None and filename_links is not None:
        file_path_probes = directory_data / filename_probes
        file_path_links = directory_data / filename_links
        graph = input_network.get_graph_from_paths(
            file_path_probes, file_path_links,
            epsilon=latency_threshold,
            clustering_distance=clustering_distance,
            ricci_curvature_alpha=ricci_curvature_alpha,
            ricci_curvature_weight_label='throughput'
        )
    else:
        raise ValueError('Need either a graphml file or two csv files as input')

    # Get the data from the networkx graph
    network = input_network.get_network_data(graph)
    graph_data, vertex_data, edge_data = network
    bounding_box = graph_data['bounding_box']
    network_coordinates = graph_data['coordinates']
    network_vertices = mesh.map_coordinates_to_support(np.array(network_coordinates), coordinates_scale, bounding_box)
    network_edges = graph_data['edges']
    network_curvatures = edge_data['ricciCurvature']

    # Setup snapshots
    if os.path.isdir(directory_output):
        shutil.rmtree(directory_output)
    os.makedirs(directory_output)

    parameters = {
        'filename_probes': str(filename_probes) if filename_probes is not None else None,
        'filename_links':str(filename_links) if filename_links is not None else None,
        'filename_graphml':str(filename_links) if filename_links is not None else None,
        'epsilon': float(latency_threshold) if latency_threshold is not None else None,
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

    if network_trim_radius is not None and not np.isposinf(network_trim_radius):
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
    # directory_links = pathlib.PurePath('toy', 'three_clusters', 'throughputs')
    # filenames_links = list(sorted(
    #     directory_links / filename
    #     for filename in os.listdir(directory_data / directory_links)
    # ))

    # count = len(filenames_links)

    # filenames_probes = [
    #     pathlib.PurePath('toy', 'three_clusters', 'probes.csv'),
    # ] * count

    directory_graphml = pathlib.PurePath('ipv4', 'graph_US')
    # filenames_graphml = [
    #     directory_graphml / filename
    #     for filename in os.listdir(directory_data / directory_graphml)
    # ]
    filenames_graphml = [
        directory_graphml / 'graph10.graphml',
        directory_graphml / 'graph22.graphml',
    ]
    count = len(filenames_graphml)

    filenames_links = [None] * count
    filenames_probes = [None] * count

    latency_thresholds = [0] * count
    clustering_distances = [None] * count

    lambdas_curvature = [1.] * count
    lambdas_smooth = [0.0002] * count
    ricci_curvature_alphas = [0.9999] * count
    initial_radii = [20.] * count
    sides = [50] * count
    mesh_scales = np.array([1.] * count)
    coordinates_scales = np.array([0.8] * count)

    network_trim_radii = 0.2 * mesh_scales

    directories_output = [
        directory_outputs / 'graph_US' \
            / f'{lambda_curvature}_{lambda_smooth}_{initial_radius}_{width}_{height}_{mesh_scale}' \
            / filename_graphml.stem
        for filename_graphml, lambda_curvature, lambda_smooth, initial_radius, width, height, mesh_scale \
            in zip(filenames_graphml, lambdas_curvature, lambdas_smooth, initial_radii, sides, sides, mesh_scales)
    ]

    maxiters = [1000] * count

    arguments = list(zip(
        filenames_probes, filenames_links,
        latency_thresholds, clustering_distances, ricci_curvature_alphas,
        lambdas_curvature, lambdas_smooth,
        initial_radii, sides, mesh_scales, coordinates_scales, network_trim_radii,
        directories_output, maxiters,
    ))
    # Need to use ProcessPoolExecutor instead of multiprocessing.Pool
    # to allow child processes to spawn their own subprocesses
    with concurrent.futures.ProcessPoolExecutor(9) as executor:
        futures = []
        for (
            filename_probes, filename_links, filename_graphml,
            latency_threshold, clustering_distance, ricci_curvature_alpha,
            lambda_curvature, lambda_smooth,
            initial_radius, side, mesh_scale, coordinates_scale, network_trim_radius,
            directory_output, maxiter
        ) in zip(
            filenames_probes, filenames_links, filenames_graphml,
            latency_thresholds, clustering_distances, ricci_curvature_alphas,
            lambdas_curvature, lambdas_smooth,
            initial_radii, sides, mesh_scales, coordinates_scales, network_trim_radii,
            directories_output, maxiters,
        ):
            future = executor.submit(
                main,
                filename_probes = filename_probes,
                filename_links = filename_links,
                filename_graphml = filename_graphml,
                latency_threshold = latency_threshold,
                clustering_distance = clustering_distance,
                ricci_curvature_alpha = ricci_curvature_alpha,
                lambda_curvature = lambda_curvature,
                lambda_smooth = lambda_smooth,
                initial_radius = initial_radius,
                sides = side,
                mesh_scale = mesh_scale,
                coordinates_scale = coordinates_scale,
                network_trim_radius = network_trim_radius,
                directory_output = directory_output,
                maxiter = maxiter,
            )
            futures.append(future)
        concurrent.futures.wait(futures)
