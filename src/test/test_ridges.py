"""Checks for the raised-ridge surface construction."""

import json
import pathlib

import numpy as np
import pytest

pytest.importorskip('torch')

from linear_geodesic_optimization import pipeline
from linear_geodesic_optimization.construction import hessian_design, ridges
from linear_geodesic_optimization.mesh.rectangle import Mesh
from linear_geodesic_optimization.optimization.torch_optimization import Computer as TorchComputer


SIDES = 80
SCALE = 1.
EPSILON = 1.01 * 2**0.5 * SCALE / SIDES
TOY_GRAPH = (
    pathlib.Path(__file__).parents[2]
    / 'data' / 'toy' / 'routing_with_volumes' / 'traffic' / 'two_clusters' / 'graph_0.json'
)


def single_edge(curvature, throughput=1., width=5 * EPSILON, height=EPSILON):
    """Return (mesh, z, kappa_G at the edge's midpoint)."""
    mesh = Mesh(SIDES, SIDES, SCALE)
    points = mesh.get_coordinates()[:, :2]
    vertices = np.array([[0.25, 0.5], [0.75, 0.5]])
    z, _, _ = ridges.build_heights(
        points, (SIDES, SIDES), vertices, [(0, 1)], [curvature], [throughput],
        width, height,
    )
    computer = TorchComputer(mesh, vertices, [(0, 1)], [curvature], EPSILON, 1., 0.)
    computer.forward(z)
    middle = np.argmin(np.linalg.norm(points - [0.5, 0.5], axis=1))
    return mesh, z, computer.kappa_G[middle]


@pytest.mark.parametrize('curvature', [-2., -0.5, 0.5, 2.])
def test_crest_curvature_matches_target(curvature):
    _, _, kappa = single_edge(curvature)
    assert np.sign(kappa) == np.sign(curvature)
    assert abs(kappa - curvature) < 0.3 * abs(curvature)


def test_heights_are_raised_and_boundary_is_flat():
    # The boundary is left free, but a ridge away from it doesn't reach it
    mesh, z, _ = single_edge(-1.)
    boundary = ~hessian_design.hessian_operators(SIDES, SCALE)[3]
    assert np.all(z >= 0.)
    assert np.all(z[boundary] == 0.)
    assert z.max() > 0.


def test_smoothing_keeps_curvature_bounded_under_refinement():
    # Without smoothing, the kink where a sagging crest meets its node's
    # dome shows up as a curvature spike that grows with resolution
    def peak_curvature(sides, smoothing):
        mesh = Mesh(sides, sides, SCALE)
        points = mesh.get_coordinates()[:, :2]
        vertices = np.array([[0.3, 0.5], [0.5, 0.5], [0.6, 0.65]])
        edges = [(0, 1), (1, 2)]
        z, _, _ = ridges.build_heights(
            points, (sides, sides), vertices, edges, [-3., -1.], [2., 1.],
            0.08, 0.004, smoothing=smoothing,
        )
        epsilon = 1.01 * 2**0.5 * SCALE / sides
        computer = TorchComputer(mesh, vertices, edges, [-3., -1.], epsilon, 1., 0.)
        computer.forward(z)
        interior = hessian_design.hessian_operators(sides, SCALE)[3]
        return np.abs(computer.kappa_G[interior]).max()

    assert peak_curvature(120, 0.01) < 1.5 * peak_curvature(60, 0.01)
    assert peak_curvature(120, 0.) > 1.5 * peak_curvature(60, 0.)


def test_negative_curvature_sags_and_positive_arches():
    mesh, z_negative, _ = single_edge(-2.)
    _, z_positive, _ = single_edge(2.)
    points = mesh.get_coordinates()[:, :2]
    middle = np.argmin(np.linalg.norm(points - [0.5, 0.5], axis=1))
    end = np.argmin(np.linalg.norm(points - [0.25, 0.5], axis=1))
    assert z_negative[middle] < z_negative[end]
    assert z_positive[middle] > z_positive[end]


@pytest.mark.parametrize('rule, expected', [
    ('max', [3., 3., 1.]),
    ('sum', [3., 4., 1.]),
    ('mean', [3., 2., 1.]),
])
def test_node_heights_rules(rule, expected):
    # Path 0 - 1 - 2 with throughputs 3 and 1
    heights = ridges.node_heights(3, [(0, 1), (1, 2)], [3., 1.], rule, 'linear', 2., 0.)
    expected = np.array(expected)
    np.testing.assert_allclose(heights, 2. * expected / expected.max())


def test_node_heights_base_and_monotonicity():
    heights = ridges.node_heights(3, [(0, 1)], [0., ], 'max', 'linear', 1., 0.25)
    np.testing.assert_allclose(heights, 0.25)  # No throughput anywhere
    throughputs = [0.5, 1., 4.]
    for transform in ('linear', 'sqrt', 'log'):
        heights = ridges.node_heights(
            6, [(0, 1), (2, 3), (4, 5)], throughputs, 'max', transform, 1., 0.25
        )
        assert heights[0] < heights[2] < heights[4]
        assert heights.max() == pytest.approx(1.)
        assert heights.min() >= 0.25


def test_build_heights_is_deterministic():
    _, z_1, _ = single_edge(-1.)
    _, z_2, _ = single_edge(-1.)
    np.testing.assert_array_equal(z_1, z_2)


def test_throughput_survives_network_loading():
    network, _, traffic = pipeline.load_network(
        filename_json=str(TOY_GRAPH), latency_threshold=4,
        clustering_distance=500000, ricci_curvature_reweight='distribution',
        ricci_curvature_distribution_reweight_scale=3.,
    )
    throughputs = network[2]['throughput']
    assert len(throughputs) == len(network[2]['ricciCurvature'])
    assert all(throughput is not None and throughput >= 0. for throughput in throughputs)
    assert max(throughputs) > 0.


def test_optimize_ridges_end_to_end(tmp_path):
    import optimization

    directory_output = tmp_path / 'run'
    optimization.optimize(
        filename_json=str(TOY_GRAPH), latency_threshold=4,
        clustering_distance=500000, ricci_curvature_reweight='distribution',
        ricci_curvature_distribution_reweight_scale=3.,
        lambda_smooth=0.005, sides=24, mesh_scale=0.25,
        directory_output=directory_output, method='ridges',
        ridge_polish_maxiter=5,
    )
    with open(directory_output / 'output.json') as file:
        output = json.load(file)
    assert output['parameters']['method'] == 'ridges'
    assert np.all(np.array(output['initial']) == 0.)
    construction = output['construction']
    assert construction['polish']['iterations'] <= 5
    assert 'fraction_below_ambient' in construction['polish']
    assert len(construction['node_heights']) == len(output['network'][0]['coordinates'])
    assert all('throughput' in edge and 'solver' in edge for edge in construction['edges'])


def test_node_heights_reference():
    # Path 0 - 1 - 2 with throughputs 2 and 1, normalized by a reference of 4
    heights = ridges.node_heights(3, [(0, 1), (1, 2)], [2., 1.], 'max', 'linear', 1., 0.2, reference=4.)
    np.testing.assert_allclose(heights, 0.2 + 0.8 * np.array([0.5, 0.5, 0.25]))
    # No reference: the run's own max maps to full height
    heights = ridges.node_heights(3, [(0, 1), (1, 2)], [2., 1.], 'max', 'linear', 1., 0.2)
    np.testing.assert_allclose(heights, 0.2 + 0.8 * np.array([1., 1., 0.5]))
    # The reference is transformed like the node values
    heights = ridges.node_heights(2, [(0, 1)], [3.], 'max', 'log', 1., 0., reference=8.)
    np.testing.assert_allclose(heights, np.log1p(3.) / np.log1p(8.))


def test_assign_ridge_height_references_uses_batch_max(tmp_path):
    import optimization
    from linear_geodesic_optimization import driver

    directory = TOY_GRAPH.parents[1] / 'reroute_link_outage'
    # graph_5 carries more traffic than graph_0
    batch = []
    for name in ['graph_0.json', 'graph_5.json']:
        arguments = driver.base_defaults() | {
            'filename_json': directory / name, 'latency_threshold': 4,
            'clustering_distance': 500000, 'ricci_curvature_reweight': 'distribution',
            'ricci_curvature_distribution_reweight_scale': 2., 'method': 'ridges',
            'sides': 16, 'mesh_scale': 0.25, 'directory_output': tmp_path / name,
        }
        batch.append(arguments)
    run_batch = [dict(arguments, ridge_height_normalization='run') for arguments in batch]
    optimization.assign_ridge_height_references([batch, run_batch])

    maxima = []
    for arguments in batch:
        network, _, _ = pipeline.load_network(**{
            key: arguments[key] for key in (
                'filename_json', 'latency_threshold', 'clustering_distance',
                'ricci_curvature_reweight', 'ricci_curvature_distribution_reweight_scale',
            )
        })
        maxima.append(ridges.node_throughputs(
            len(network[0]['coordinates']), network[0]['edges'], network[2]['throughput'], 'max'
        ).max())
    assert maxima[0] < maxima[1]
    assert [arguments['ridge_height_reference'] for arguments in batch] == pytest.approx([maxima[1]] * 2)
    assert all(arguments['ridge_height_reference'] is None for arguments in run_batch)

    # The lower-traffic snapshot's tallest node stays below full height
    optimization.optimize(**batch[0])
    with open(tmp_path / 'graph_0.json' / 'output.json') as file:
        output = json.load(file)
    full_height = 0.05 * 0.3 * 0.25
    assert max(output['construction']['node_heights']) < 0.9 * full_height
    assert output['parameters']['ridge_height_reference'] == pytest.approx(maxima[1])
