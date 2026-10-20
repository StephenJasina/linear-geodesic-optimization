"""
Building blocks shared by the methods in optimization.py: reading the
input network, mapping it onto the mesh, running L-BFGS on a loss, and
writing output.json.
"""

import json
import os
import shutil
import typing

import networkx as nx
import numpy as np
import numpy.typing as npt
import scipy.optimize

from linear_geodesic_optimization.data import input_network


def load_network(
    *,
    filename_probes=None,
    filename_links=None,
    filename_graphml=None,
    filename_json=None,
    latency_threshold=None,
    clustering_distance=None,
    ricci_curvature_alpha=0.,
    ricci_curvature_reweight=None,
    ricci_curvature_distribution_reweight_scale=0.,
):
    """
    Read the input network and compute its Ricci curvatures.

    Return `(network, routes, traffic)`, where `network` is the
    `(graph_data, vertex_data, edge_data)` triple from
    `input_network.get_network_data`.
    """
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
            ricci_curvature_alpha=ricci_curvature_alpha,
            ricci_curvature_reweight=ricci_curvature_reweight,
            ricci_curvature_distribution_reweight_scale=ricci_curvature_distribution_reweight_scale,
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

    return input_network.get_network_data(graph), routes, traffic

def map_network_to_mesh(mesh, network, coordinates_scale):
    """
    Map a network's coordinates into the mesh's support.

    Return `(network_vertices, network_edges, network_curvatures)`.
    """
    graph_data, _, edge_data = network
    network_vertices = mesh.map_coordinates_to_support(
        np.array(graph_data['coordinates']), coordinates_scale,
        graph_data['bounding_box']
    )
    return network_vertices, graph_data['edges'], edge_data['ricciCurvature']

def fat_edge_width(sides: int, mesh_scale: float) -> float:
    """
    Return the width of the fat edges used by the curvature loss: just
    over one mesh diagonal.
    """
    return 1.01 * 2**0.5 * mesh_scale / sides

def reset_output_directory(directory_output) -> None:
    """Create `directory_output`, emptying it first if it exists."""
    if os.path.isdir(directory_output):
        shutil.rmtree(directory_output)
    os.makedirs(directory_output)

def get_computer_type(backend: typing.Literal['legacy', 'torch']):
    """
    Return the loss `Computer` class for the given backend. The torch
    backend is imported only when requested, so that the legacy backend
    works without PyTorch installed.
    """
    if backend == 'legacy':
        from linear_geodesic_optimization.optimization.optimization import Computer
        return Computer
    if backend == 'torch':
        from linear_geodesic_optimization.optimization.torch_optimization import Computer
        return Computer
    raise ValueError(f'Unknown optimization backend: {backend}')

def base_parameters(
    *,
    latency_threshold,
    clustering_distance,
    ricci_curvature_alpha,
    ricci_curvature_reweight,
    ricci_curvature_distribution_reweight_scale,
    lambda_curvature,
    lambda_smooth,
    smooth_loss_type,
    initial_radius,
    sides,
    mesh_scale,
    coordinates_scale,
    method,
    backend,
    network_trim_radius,
) -> dict[str, typing.Any]:
    """
    Return the parameters recorded in output.json that every method
    shares. Methods add their own parameters to this dictionary.
    """
    return {
        'epsilon': float(latency_threshold) if latency_threshold is not None else None,
        'clustering_distance': float(clustering_distance) if clustering_distance is not None else None,
        'ricci_curvature_alpha': float(ricci_curvature_alpha) if ricci_curvature_alpha is not None else None,
        'ricci_curvature_reweight': ricci_curvature_reweight,
        'ricci_curvature_distribution_reweight_scale': ricci_curvature_distribution_reweight_scale,
        'lambda_curvature': float(lambda_curvature),
        'lambda_smooth': float(lambda_smooth),
        'smooth_loss_type': smooth_loss_type,
        # None (an infinite radius) starts flat; collation.py then uses
        # 'initial' as the baseline
        'initial_radius': float(initial_radius) if initial_radius is not None else None,
        'width': int(sides),
        'height': int(sides),
        'mesh_scale': float(mesh_scale),
        'coordinates_scale': float(coordinates_scale),
        'method': method,
        'backend': backend,
        'network_trim_radius': float(network_trim_radius) if network_trim_radius is not None else None,
    }

def minimize_lbfgs(
    computer,
    z_0: npt.NDArray[np.float64],
    options: typing.Optional[dict] = None,
    initial_diagnostics_z: typing.Optional[npt.NDArray[np.float64]] = None,
) -> scipy.optimize.OptimizeResult:
    """
    Minimize `computer`'s loss with L-BFGS-B, starting from `z_0`.

    The computer's diagnostics are called once with
    `initial_diagnostics_z` before the first iteration (where None
    reports the mesh's current state), and then after every iteration.
    """
    computer.diagnostics(initial_diagnostics_z)
    return scipy.optimize.minimize(
        computer.forward, z_0, method='L-BFGS-B', jac=computer.reverse,
        callback=computer.diagnostics,
        options=options,
    )

def write_output(
    directory_output,
    *,
    parameters,
    initial,
    final,
    network,
    routes,
    traffic,
    **extra,
) -> None:
    """
    Write output.json, the file read by collation.py and by the next
    run of a sequentially initialized batch. Keyword arguments in
    `extra` are written as additional top-level keys.
    """
    with open(directory_output / 'output.json', 'w') as file_output:
        json.dump({
            'parameters': parameters,
            'initial': initial.tolist(),
            'final': final.tolist(),
            'network': network,
            'routes': routes,
            'traffic': traffic,
            **extra,
        }, file_output, ensure_ascii=False, indent=4)
