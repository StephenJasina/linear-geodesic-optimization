import argparse
import itertools
import json
import os
import pathlib
import shutil
import warnings

import networkx as nx
import numpy as np
import scipy

import linear_geodesic_optimization.batch as batch
from linear_geodesic_optimization.data import input_network
from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh
from linear_geodesic_optimization.optimization import optimization


# Error on things like division by 0
warnings.simplefilter('error')

# List of parameters that correspond to extant files. This is used to
# help generate output directory names (by removing extensions when they
# would be unneeded) and determining actual locations of input files (by
# prepending ../data)
parameter_name_filenames = [
    'filename_probes',
    'filename_links',
    'filename_graphml',
    'filename_json',
]

def argument_to_string(
    arguments,
    parameter_name
):
    if parameter_name in parameter_name_filenames:
        return pathlib.PurePath(arguments[parameter_name]).stem
    else:
        return str(arguments[parameter_name])

def optimize(
    *,  # All parameters are keyword only
    filename_probes=None,
    filename_links=None,
    filename_graphml=None,
    filename_json=None,
    latency_threshold=None,
    clustering_distance=None,
    ricci_curvature_alpha=0.9999,
    lambda_curvature=1.,
    lambda_smooth=0.,
    initial_radius,
    sides,
    mesh_scale=1.,
    coordinates_scale=0.8,
    network_trim_radius=None,
    directory_output,
    maxiter=None,
    initialization_file_path=None,
    **kwargs
):
    # Construct the mesh
    width = height = sides
    mesh = RectangleMesh(width, height, mesh_scale)

    # Construct the networkx graph
    routes = None
    traffic = None
    if filename_graphml is not None:
        graph = nx.read_graphml(filename_graphml)
    elif filename_json is not None:
        file_path_json = filename_json
        graph, routes, traffic = input_network.get_graph_from_json(
            file_path_json,
            epsilon=latency_threshold,
            clustering_distance=clustering_distance,
            return_traffic=True,
        )
    elif filename_probes is not None and filename_links is not None:
        file_path_probes = filename_probes
        file_path_links = filename_links
        graph = input_network.get_graph_from_csvs(
            file_path_probes, file_path_links,
            epsilon=latency_threshold,
            clustering_distance=clustering_distance,
            ricci_curvature_alpha=ricci_curvature_alpha,
        )
    else:
        raise ValueError('Need a graphml file, a json file, or two csv files as input')

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
        'filename_links': str(filename_links) if filename_links is not None else None,
        'filename_graphml': str(filename_graphml) if filename_graphml is not None else None,
        'epsilon': float(latency_threshold) if latency_threshold is not None else None,
        'clustering_distance': float(clustering_distance) if clustering_distance is not None else None,
        'ricci_curvature_alpha': float(ricci_curvature_alpha),
        'lambda_curvature': float(lambda_curvature),
        'lambda_smooth': float(lambda_smooth),
        'initial_radius': float(initial_radius),
        'width': int(width),
        'height': int(height),
        'mesh_scale': float(mesh_scale),
        'coordinates_scale': float(coordinates_scale),
        'network_trim_radius': float(network_trim_radius) if network_trim_radius is not None else None,
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
            'routes': routes,
            'traffic': traffic,
        }, file_output, ensure_ascii=False, indent=4)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    args = parser.parse_args()
    config_file = pathlib.PurePath(args.config_file)

    # Read JSON config file and generate optimization parameters from it
    defaults = {
        'filename_probes': None,
        'filename_links': None,
        'filename_graphml': None,
        'filename_json': None,
        'latency_threshold': None,
        'clustering_distance': None,
        'ricci_curvature_alpha': 0.,
        'lambda_curvature': 1.,
        'lambda_smooth': 0.,
        'initial_radius': 20.,
        'sides': 50,
        'mesh_scale': 1.,
        'coordinates_scale': 0.8,
        'network_trim_radius': None,
        'maxiter': None,
        'initialization_file_path': None,
        'index': None # Additional unique ID
    }
    with open(config_file, 'r') as f:
        arguments, settings = batch.parse_json(f, defaults)

    # Determine where the outputs are going
    if 'output_format' in settings:
        output_format = settings['output_format']
    else:
        output_format = ['filename_json']
    # Check that the output format has the right types
    if not isinstance(output_format, list):
        raise TypeError('Output format must be a list of (strings or lists of strings)')
    for output_format_part in output_format:
        if isinstance(output_format_part, str):
            if output_format_part not in defaults:
                raise ValueError(f'{output_format_part} not a valid part of output format')
        elif isinstance(output_format_part, list):
            for output_format_part_part in output_format_part:
                if isinstance(output_format_part_part, str):
                    if output_format_part_part not in defaults:
                        raise ValueError(f'{output_format_part_part} not a valid part of output format')
                else:
                    raise TypeError('Output format must be a list of (strings or lists of strings)')
        else:
            raise TypeError('Output format must be a list of (strings or lists of strings)')
    # Generate the output directories from the given format. Check if
    # they already exist
    for index, argument_dict in enumerate(arguments):
        argument_dict['index'] = index
        # Also prepend ../outputs (which is the path of the outputs
        # directory relative to the script)
        directory_output = pathlib.PurePath('..', 'outputs') / pathlib.PurePath(*(settings['directory_output'] + [
            argument_to_string(argument_dict, output_format_part) if isinstance(output_format_part, str) else
            '_'.join([argument_to_string(argument_dict, output_format_part_part) for output_format_part_part in output_format_part])
            for output_format_part in output_format
        ]))
        if os.path.exists(directory_output):
            raise ValueError(f'{str(directory_output)} already exists')
        argument_dict['directory_output'] = directory_output
    # Check whether the output directories overlap with themselves
    # TODO: This can probably be improved by not constructing a set
    if len(set(str(argument_dict['directory_output']) for argument_dict in arguments)) != len(arguments):
        raise ValueError('Some output directories are duplicated')

    # Optionally prepend a directory to where the input files are
    # stored. This is to allow different choices of where the paths are
    # relative to (the main data directory, the script's current
    # directory, or the config file's directory (default))
    directory_data_type = (
        'config' if 'directory_data_type' not in settings else
        settings['directory_data_type']
    )
    directory_data_parent = (
        pathlib.PurePath('..', 'data') if directory_data_type == 'data' else
        pathlib.PurePath() if directory_data_type == 'script' else
        config_file.parent
    )
    directory_data = directory_data_parent / (pathlib.PurePath(*settings['directory_data']) if 'directory_data' in settings else '')
    for argument_dict in arguments:
        for parameter_name in parameter_name_filenames:
            if argument_dict[parameter_name] is not None:
                argument_dict[parameter_name] = directory_data / argument_dict[parameter_name]

    if not arguments or 'dry_run' in settings and settings['dry_run']:
        # Exit before producing output
        return

    # Determine the initialization strategy, and then use that to run
    # the optimization, using multiprocessing where possible.
    if 'initialization' in settings:
        initialization = settings['initialization']
    else:
        initialization = 'sphere'
    n_cores = settings['n_cores'] if 'n_cores' in settings else None
    if initialization == 'sphere':
        for argument_dict in arguments:
            argument_dict['initialization_file_path'] = None
        batch.run_multiprocessed(optimize, arguments, n_cores)
    elif initialization == 'sequential':
        # TODO: This behaves weirdly if the arguments don't form a
        # single sequence (e.g., optimizing over a space of many input
        # files and hyperparameter selections simultaneously)
        arguments[0]['initialization_file_path'] = None
        for argument_dict_previous, argument_dict in itertools.pairwise(arguments):
            argument_dict['initialization_file_path'] = argument_dict_previous['directory_output'] / 'output.json'
        batch.run_sequential(optimize, arguments)
    elif initialization == 'first':
        optimize(**arguments[0])
        initializaiton_file_path = arguments[0]['directory_output'] / 'output.json'
        for argument_dict in arguments[1:]:
            argument_dict['initialization_file_path'] = initializaiton_file_path
        batch.run_multiprocessed(optimize, arguments[1:], n_cores)
    else:
        raise ValueError(f'Invalid intiaization strategy "{initialization}"')

if __name__ == '__main__':
    main()
