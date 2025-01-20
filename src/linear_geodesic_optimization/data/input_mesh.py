import os
import pickle
import typing

import numpy as np

from linear_geodesic_optimization.graph import convex_hull
from linear_geodesic_optimization.data import input_network
from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh


def get_mesh_from_directory(
        directory: str,
        max_iterations: typing.Optional[int] = None,
        postprocessed: bool = False,
        initialization_path: typing.Optional[str] = None
) -> RectangleMesh:
    """
    Given a directory of output, return a mesh of the final output.

    This function can be used to get the height map from precomputed
    output that has been pickled. Some extra postprocessing is
    optionally done to make the output a bit more aesthetically
    pleasing.
    """
    with open(os.path.join(directory, 'parameters'), 'rb') as f:
        parameters = pickle.load(f)

        probes_file_path = os.path.join('..', 'data', parameters['filename_probes'])
        latencies_file_path = os.path.join('..', 'data', parameters['filename_links'])
        epsilon = parameters['epsilon']
        clustering_distance = parameters['clustering_distance']
        should_remove_TIVs = parameters['should_remove_TIVs']
        width = parameters['width']
        height = parameters['height']

    if initialization_path is None:
        initialization_path = os.path.join(directory, '0')
    with open(initialization_path, 'rb') as f:
        iteration_data = pickle.load(f)
        z_0 = np.array(iteration_data['mesh_parameters'])

    iteration = max(int(name)
                    for name in os.listdir(directory)
                    if name.isdigit())
    if max_iterations is not None:
        iteration = min(iteration, max_iterations)

    with open(os.path.join(directory, str(iteration)), 'rb') as f:
        iteration_data = pickle.load(f)
        z = np.array(iteration_data['mesh_parameters'])

    graph = input_network.get_graph_from_paths(
        probes_file_path, latencies_file_path,
        epsilon=epsilon,
        clustering_distance=clustering_distance,
        should_remove_tivs=should_remove_TIVs
    )
    network_data = input_network.get_network_data(graph)

    return get_mesh(z, width, height, network_data, postprocessed, z_0)

def get_mesh(
        z: typing.List[np.float64],
        width: int,
        height: int,
        network_data,
        coordinates_scale: float,
        mesh_scale: float = 1.,
        postprocessed: bool = False,
        z_0: typing.Optional[typing.List[np.float64]] = None,
        network_trim_radius: np.float64 = np.inf,
        z_hole = -0.5,
        mesh: typing.Optional[RectangleMesh] = None
) -> RectangleMesh:
    """
    Return a mesh with the given parameters.

    Some extra postprocessing is optionally done to make the output a
    bit more aesthetically pleasing.
    """
    if mesh is None:
        mesh = RectangleMesh(width, height, mesh_scale)

    graph_data, vertex_data, edge_data = network_data

    coordinates = np.array(graph_data['coordinates'])
    bounding_box = graph_data['bounding_box']
    network_edges = graph_data['edges']
    network_vertices = mesh.map_coordinates_to_support(coordinates, coordinates_scale, bounding_box)

    mesh.trim_to_graph(network_vertices, network_edges, network_trim_radius)

    if postprocessed:
        network_convex_hulls = convex_hull.compute_connected_convex_hulls(
            network_vertices, network_edges)
        vertices = mesh.get_coordinates()
        distances = np.array([
            convex_hull.distance_to_convex_hulls(
                np.array(vertex_coordinate),
                network_vertices,
                network_convex_hulls
            )
            for vertex_coordinate in vertices[:, :2]
        ])
        # Add a small amount of space around the convex hull
        distances = np.maximum(distances - 0.05, 0.)
        z = z - np.array(z_0)
        z = (z - np.amin(z[distances == 0.], initial=np.amin(z))) \
            * np.exp(-1000 * distances**2)
        z = z - np.amin(z)
        z = z * 0.15 / np.amax(z)

    mesh.set_parameters(z)
    return mesh
