import os
import pickle
import typing

import numpy as np

from linear_geodesic_optimization import convex_hull
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

        probes_file_path = os.path.join('..', 'data', parameters['probes_filename'])
        latencies_file_path = os.path.join('..', 'data', parameters['latencies_filename'])
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

    network, latencies = input_network.get_graph(
        probes_file_path, latencies_file_path,
        epsilon, clustering_distance, should_remove_TIVs,
        should_include_latencies=True
    )
    network = input_network.extract_from_graph(network, latencies)

    return get_mesh(z, width, height, network, postprocessed, z_0)

def get_mesh(
        z: typing.List[np.float64],
        width: int,
        height: int,
        network,
        coordinates_scale: float,
        postprocessed: bool = False,
        z_0: typing.Optional[typing.List[np.float64]] = None,
) -> RectangleMesh:
    """
    Return a mesh with the given parameters.

    Some extra postprocessing is optionally done to make the output a
    bit more aesthetically pleasing.
    """
    mesh = RectangleMesh(width, height)

    coordinates, bounding_box, network_edges, _, _ = network
    coordinates = np.array(coordinates)
    network_vertices = mesh.map_coordinates_to_support(coordinates, coordinates_scale, bounding_box)
    nearest_vertex_indices = [mesh.nearest_vertex(network_vertex).index()
                              for network_vertex in network_vertices]
    network_convex_hulls = convex_hull.compute_connected_convex_hulls(
        network_vertices, network_edges)

    if postprocessed:
        vertices = mesh.get_coordinates()
        x = list(sorted(set(vertices[:,0])))
        y = list(sorted(set(vertices[:,1])))
        distances = np.array([
            convex_hull.distance_to_convex_hulls(
                np.array([px, py]),
                network_vertices,
                network_convex_hulls
            )
            for px in x
            for py in y
        ])
        # Add a small amount of space around the convex hull
        distances = np.maximum(distances - 0.05, 0.)
        z = z - np.array(z_0)
        z = (z - np.amin(z[distances == 0.], initial=np.amin(z))) \
            * np.exp(-1000 * distances**2)
        z = z - np.amin(z)

    mesh.set_parameters(z)
    return mesh
