import argparse
import collections
import functools
import itertools
import json
import pathlib
import sys

import networkx as nx
import numpy as np
import pygeodesic.geodesic as pgeo
from scipy import stats

sys.path.insert(0, str(pathlib.PurePath('.')))
import linear_geodesic_optimization.driver as driver
from linear_geodesic_optimization.data import input_network
from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh


def compute_edge_geodesic_lengths(mesh, network_vertices, network_edges):
    """
    Compute the exact geodesic length between the endpoints of each
    network edge on the given mesh.

    This follows the same vertex-insertion approach as
    `collation.compute_geodesics_from_graph` (its `exact_geodesics=True`
    branch), but returns the solver's own arc lengths directly instead
    of reconstructing a 2D-projected, mesh-scale-divided path (which
    would drop the height coordinate and thus distort the length).
    """
    network_indices = set(itertools.chain.from_iterable(network_edges))
    bad_indices = set()
    for index in network_indices:
        try:
            mesh.add_vertex_at_coordinates(network_vertices[index], 0.0001)
        except ValueError:
            bad_indices.add(index)

    coordinates = mesh.get_coordinates()
    topology = mesh.get_topology()
    faces = np.array([
        [vertex.index for vertex in face.vertices()]
        for face in topology.faces()
    ])
    network_vertex_indices_to_mesh_vertex_indices = [
        mesh.nearest_vertex(network_vertex).index
        for network_vertex in network_vertices
    ]

    path_solver = pgeo.PyGeodesicAlgorithmExact(coordinates, faces)

    lengths = []
    for u_index, v_index in network_edges:
        if u_index in bad_indices or v_index in bad_indices:
            lengths.append(None)
            continue
        distance, _ = path_solver.geodesicDistance(
            network_vertex_indices_to_mesh_vertex_indices[u_index],
            network_vertex_indices_to_mesh_vertex_indices[v_index]
        )
        lengths.append(float(distance))

    mesh.remove_added_vertices()
    mesh.restore_removed_vertices()

    return lengths


@functools.lru_cache(maxsize=None)
def _load_input_network(
    filename_graphml, filename_json, filename_probes, filename_links,
    latency_threshold, clustering_distance
):
    """
    Rebuild the network graph from whichever input file the config
    points to (mirroring `optimization.optimize`'s graph-construction
    logic), and return its `get_network_data` decomposition. Cached
    since many outputs in a group (e.g. ones only differing in `sides`
    or `lambda_smooth`) share the same input file and thresholding.
    """
    if filename_graphml is not None:
        graph = nx.read_graphml(filename_graphml)
    elif filename_json is not None:
        graph = input_network.get_graph_from_json(
            filename_json,
            epsilon=latency_threshold,
            clustering_distance=clustering_distance,
            should_compute_curvatures=False,
        )
    elif filename_probes is not None and filename_links is not None:
        graph = input_network.get_graph_from_csvs(
            filename_probes, filename_links,
            epsilon=latency_threshold,
            clustering_distance=clustering_distance,
            should_compute_curvatures=False,
        )
    else:
        raise ValueError(
            'Need a graphml file, a json file, or two csv files as input'
        )
    return input_network.get_network_data(graph)


def get_edge_rtts(argument_dict, network_edges, labels):
    """
    Return a list of per-edge RTTs, aligned with `network_edges`/
    `labels`, read from the input network file specified by the
    config (rather than from `output.json`, which older outputs never
    persisted RTT into). Matching is done by label pair rather than
    index, so it's robust to any incidental reordering between this
    freshly rebuilt graph and the one baked into the output.
    """
    input_graph_data, _, input_edge_data = _load_input_network(
        argument_dict['filename_graphml'],
        argument_dict['filename_json'],
        argument_dict['filename_probes'],
        argument_dict['filename_links'],
        argument_dict['latency_threshold'],
        argument_dict['clustering_distance'],
    )
    if 'rtt' not in input_edge_data:
        raise ValueError(
            f"{argument_dict['directory_output']}'s input has no per-edge "
            "RTT data to fit against (its network was likely built from a "
            "plain GraphML file rather than RTT-labeled CSV/JSON input)"
        )

    rtt_by_label_pair = {
        (input_graph_data['labels'][u], input_graph_data['labels'][v]): rtt
        for (u, v), rtt in zip(input_graph_data['edges'], input_edge_data['rtt'])
    }
    return [
        rtt_by_label_pair.get((labels[u], labels[v]))
        for u, v in network_edges
    ]


def process_output(argument_dict):
    """
    Load a single optimization output and compute, for every network
    edge, its geodesic length on the fitted mesh, along with a
    least-squares fit of geodesic length as a function of RTT.
    """
    directory_output = argument_dict['directory_output']
    with open(directory_output / 'output.json') as file_output:
        output = json.load(file_output)

    parameters = output['parameters']
    graph_data, vertex_data, edge_data = output['network']

    mesh = RectangleMesh(
        parameters['width'], parameters['height'], parameters['mesh_scale']
    )
    mesh.set_parameters(np.array(output['final']))

    network_vertices = mesh.map_coordinates_to_support(
        np.array(graph_data['coordinates']),
        parameters['coordinates_scale'],
        graph_data['bounding_box']
    )
    network_edges = graph_data['edges']
    labels = graph_data['labels']
    rtts = get_edge_rtts(argument_dict, network_edges, labels)
    curvatures = edge_data['ricciCurvature']

    lengths = compute_edge_geodesic_lengths(mesh, network_vertices, network_edges)

    valid_indices = [
        i for i in range(len(network_edges))
        if rtts[i] is not None and lengths[i] is not None
    ]
    fit = stats.linregress(
        [rtts[i] for i in valid_indices],
        [lengths[i] for i in valid_indices]
    )

    return {
        'epsilon': parameters['epsilon'],
        'labels': labels,
        'edges': network_edges,
        'curvatures': curvatures,
        'lengths': lengths,
        'fit': fit,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    args = parser.parse_args()
    config_file = pathlib.PurePath(args.config_file)

    arguments, settings, defaults = driver.load_config(config_file)
    output_format = driver.get_output_format(settings, defaults)
    driver.assign_output_directories(
        arguments, settings, output_format, existence_check='must_exist'
    )

    groups_by_threshold = collections.defaultdict(list)
    for argument_dict in itertools.chain.from_iterable(arguments):
        result = process_output(argument_dict)
        groups_by_threshold[result['epsilon']].append(result)

    for epsilon, group in sorted(
        groups_by_threshold.items(), key=lambda item: (item[0] is None, item[0])
    ):
        print(f'Latency threshold: {epsilon} ({len(group)} outputs)')

        # Curvature is a property of the network graph, which is shared
        # across every output in this group (they were all built with
        # the same latency threshold), so any one output can be used to
        # identify the negative-curvature edges.
        representative = group[0]
        negative_curvature_edge_indices = [
            i for i, curvature in enumerate(representative['curvatures'])
            if curvature is not None and curvature < 0
        ]

        for edge_index in negative_curvature_edge_indices:
            u_index, v_index = representative['edges'][edge_index]
            label_pair = (
                representative['labels'][u_index],
                representative['labels'][v_index]
            )

            rtt_equivalents = []
            for result in group:
                edge_indices_by_label_pair = {
                    (result['labels'][u], result['labels'][v]): i
                    for i, (u, v) in enumerate(result['edges'])
                }
                if label_pair not in edge_indices_by_label_pair:
                    continue
                length = result['lengths'][edge_indices_by_label_pair[label_pair]]
                slope, intercept = result['fit'].slope, result['fit'].intercept
                if length is None or slope == 0:
                    continue
                rtt_equivalents.append((length - intercept) / slope)

            if len(rtt_equivalents) < 2:
                continue

            mean_rtt_equivalent = sum(rtt_equivalents) / len(rtt_equivalents)
            relative_change = (
                (max(rtt_equivalents) - min(rtt_equivalents)) / mean_rtt_equivalent
            )
            print(
                f'  Edge {label_pair}: relative change in RTT-normalized '
                f'geodesic length = {relative_change:.4f} '
                f'(based on {len(rtt_equivalents)}/{len(group)} outputs) '
                f'({rtt_equivalents})'
            )


if __name__ == '__main__':
    main()
