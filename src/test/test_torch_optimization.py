"""Numerical equivalence checks for the array-based optimizer."""

import json

import numpy as np
import pytest

pytest.importorskip('torch')

from linear_geodesic_optimization.mesh.rectangle import Mesh
from linear_geodesic_optimization.optimization.optimization import Computer as LegacyComputer
from linear_geodesic_optimization.optimization.torch_optimization import Computer as TorchComputer


@pytest.mark.parametrize('sides', [6, 12, 20])
@pytest.mark.parametrize('lambda_smooth', [0.0, 0.005])
@pytest.mark.parametrize(
    'curvatures',
    [
        [0.1, -0.2, 0.05, 0.3, None],
        [0.0, 0.0, 0.0, 0.0, None],
        [0.3, 0.15, 0.25, 0.4, 0.2],
        [-0.4, -0.2, -0.3, -0.1, -0.25],
    ],
    ids=['mixed_with_missing', 'flat', 'positive', 'negative'],
)
def test_loss_and_gradient_match_legacy(sides, lambda_smooth, curvatures):
    mesh = Mesh(sides, sides, 0.25)
    vertices = np.array([
        [0.025, 0.025], [0.225, 0.025],
        [0.225, 0.225], [0.025, 0.225],
    ])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    epsilon = 1.01 * 2**0.5 * 0.25 / sides
    legacy = LegacyComputer(
        mesh, vertices, edges, curvatures, epsilon, 1.0, lambda_smooth
    )
    fast = TorchComputer(
        mesh, vertices, edges, curvatures, epsilon, 1.0, lambda_smooth
    )

    for z in (
        0.01 * np.sin(np.arange(sides * sides)),
        0.0001 * np.random.default_rng(7).normal(size=sides * sides),
    ):
        mesh.set_parameters(z)
        expected_loss = legacy.forward()
        expected_gradient = legacy.reverse()
        np.testing.assert_allclose(fast.forward(z), expected_loss, rtol=1e-9)
        np.testing.assert_allclose(
            fast.reverse(z), expected_gradient, rtol=1e-7, atol=1e-7
        )


def test_gradient_matches_directional_difference():
    sides = 8
    mesh = Mesh(sides, sides, 0.25)
    vertices = np.array([[0.02, 0.02], [0.23, 0.20]])
    computer = TorchComputer(
        mesh, vertices, [(0, 1)], [0.1],
        1.01 * 2**0.5 * 0.25 / sides, 1.0, 0.005,
    )
    rng = np.random.default_rng(19)
    z = rng.normal(0, 0.0002, sides * sides)
    direction = rng.normal(size=sides * sides)
    direction /= np.linalg.norm(direction)
    step = 1e-7
    analytic = computer.reverse(z) @ direction
    numerical = (
        computer.forward(z + step * direction)
        - computer.forward(z - step * direction)
    ) / (2 * step)
    np.testing.assert_allclose(analytic, numerical, rtol=3e-4, atol=1e-5)


def test_diagnostics_write_compatible_snapshot(tmp_path):
    mesh = Mesh(6, 6, 0.25)
    vertices = np.array([[0.02, 0.02], [0.23, 0.20]])
    computer = TorchComputer(
        mesh, vertices, [(0, 1)], [0.1],
        1.01 * 2**0.5 * 0.25 / 6, 1.0, 0.005,
        directory=tmp_path,
    )
    z = 0.01 * np.sin(np.arange(36))
    computer.diagnostics(z)
    with open(tmp_path / '0.json') as snapshot:
        data = json.load(snapshot)
    np.testing.assert_allclose(data['mesh_parameters'], z)
    assert np.isfinite(data['L_curvature'])
    assert np.isfinite(data['L_smooth'])
