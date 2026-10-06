"""Checks for the Hessian-field surface construction."""

import numpy as np
import pytest

pytest.importorskip('torch')

from linear_geodesic_optimization.construction import hessian_design
from linear_geodesic_optimization.mesh.rectangle import Mesh
from linear_geodesic_optimization.optimization.torch_optimization import Computer as TorchComputer


SIDES = 40
SCALE = 0.25
EPSILON = 1.01 * 2**0.5 * SCALE / SIDES


def construct(vertices, edges, curvatures, **kwargs):
    """Return (heights, mean kappa_G per edge) after calibration."""
    mesh = Mesh(SIDES, SIDES, SCALE)
    z = hessian_design.solve_heights(
        mesh, vertices, edges, curvatures, EPSILON, **kwargs
    )
    computer = TorchComputer(
        mesh, vertices, edges, curvatures, EPSILON, 1.0, 0.0
    )
    z = hessian_design.calibrate_scale(computer.forward, z) * z
    computer.forward(z)
    residuals = hessian_design.edge_residuals(
        computer.kappa_G, mesh.get_coordinates()[:, :2],
        hessian_design.hessian_operators(SIDES, SCALE)[3],
        vertices, edges, curvatures, EPSILON
    )
    return z, [residual['mean_kappa_G'] for residual in residuals]


def test_operators_are_exact_on_quadratics():
    sides = 12
    dxx, dyy, dxy, interior = hessian_design.hessian_operators(sides, SCALE)
    xy = Mesh(sides, sides, SCALE).get_coordinates()[:, :2]
    x, y = xy[:, 0], xy[:, 1]
    f = 0.3 * x**2 + 0.7 * x * y - 1.1 * y**2
    assert interior.sum() == (sides - 2)**2
    np.testing.assert_allclose(dxx @ f, 0.6, atol=1e-10)
    np.testing.assert_allclose(dyy @ f, -2.2, atol=1e-10)
    np.testing.assert_allclose(dxy @ f, 0.7, atol=1e-10)


@pytest.mark.parametrize('curvature', [-1.0, -0.3, 0.3, 1.0])
@pytest.mark.parametrize('anisotropy', [1.0, 2.0])
def test_edge_target_determinant(curvature, anisotropy):
    target = hessian_design.edge_target(
        np.array([0.1, 0.2]), np.array([0.3, 0.25]), curvature, anisotropy
    )
    t_xx, t_yy, t_xy = target
    assert t_xx * t_yy - t_xy**2 == pytest.approx(curvature)


def test_saddle_rises_along_edge():
    a, b = np.array([0., 0.]), np.array([1., 0.])
    t_xx, t_yy, _ = hessian_design.edge_target(a, b, -1.0)
    assert t_xx > 0 > t_yy
    t_xx, t_yy, _ = hessian_design.edge_target(a, b, -1.0, orientation='basin')
    assert t_xx < 0 < t_yy


@pytest.mark.parametrize('curvature', [-1.0, -0.25, 0.25, 1.0])
def test_single_edge_curvature_is_matched(curvature):
    vertices = np.array([[0.08, 0.125], [0.17, 0.125]])
    _, (mean_kappa_G,) = construct(vertices, [(0, 1)], [curvature])
    assert mean_kappa_G == pytest.approx(curvature, rel=1e-2)


def test_basin_negates_hill():
    vertices = np.array([[0.08, 0.125], [0.17, 0.125]])
    mesh = Mesh(SIDES, SIDES, SCALE)
    hill = hessian_design.solve_heights(mesh, vertices, [(0, 1)], [-1.0], EPSILON)
    basin = hessian_design.solve_heights(
        mesh, vertices, [(0, 1)], [-1.0], EPSILON, orientation='basin'
    )
    np.testing.assert_allclose(basin, -hill)


def test_two_clusters_signs_and_loss():
    vertices = np.array([
        [0.04, 0.10], [0.04, 0.15], [0.09, 0.125],
        [0.16, 0.125], [0.21, 0.10], [0.21, 0.15],
    ])
    edges = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)]
    curvatures = [0.5] * 6 + [-1.0]
    z, means = construct(vertices, edges, curvatures)
    assert all(mean > 0 for mean in means[:6])
    assert means[6] < 0

    mesh = Mesh(SIDES, SIDES, SCALE)
    computer = TorchComputer(mesh, vertices, edges, curvatures, EPSILON, 1.0, 0.0)
    assert computer.forward(z) < 0.5 * computer.forward(np.zeros_like(z))


@pytest.mark.parametrize('curvature', [-2.0, -0.5, 0.0, 0.5, 2.0])
def test_projection_is_nearest_with_determinant(curvature):
    rng = np.random.default_rng(0)
    hessians = rng.normal(size=(50, 3))
    projected = hessian_design.project_to_determinant(
        hessians, np.full(len(hessians), curvature)
    )
    determinants = projected[:, 0] * projected[:, 1] - projected[:, 2]**2
    np.testing.assert_allclose(determinants, curvature, atol=1e-9)

    # Compare against a brute-force search over matrices sharing the
    # input's eigenvectors, parametrizing the hyperbola mu_1 mu_2 = kappa
    def frobenius_squared(a, b):
        d = a - b
        return d[..., 0]**2 + d[..., 1]**2 + 2 * d[..., 2]**2

    for hessian, best in zip(hessians, projected):
        matrix = np.array([[hessian[0], hessian[2]], [hessian[2], hessian[1]]])
        _, q = np.linalg.eigh(matrix)
        if curvature == 0.:
            mus = [(m, 0.) for m in np.linspace(-5, 5, 2001)] \
                + [(0., m) for m in np.linspace(-5, 5, 2001)]
        else:
            r = np.sqrt(abs(curvature))
            mus = [
                (sign * r * np.exp(s), np.sign(curvature) * sign * r * np.exp(-s))
                for s in np.linspace(-5, 5, 4001) for sign in (-1., 1.)
            ]
        candidates = np.array([
            (m[0, 0], m[1, 1], m[0, 1])
            for mu in mus for m in (q @ np.diag(mu) @ q.T,)
        ])
        assert frobenius_squared(best, hessian) \
            <= frobenius_squared(candidates, hessian).min() + 1e-6


def test_iterations_resolve_conflicting_targets():
    # Perpendicular spokes of a hub ask for opposite saddles at the hub,
    # so the designed targets cancel there and need to be reoriented
    vertices = np.array([
        [0.125, 0.125], [0.2, 0.125], [0.125, 0.2],
        [0.05, 0.125], [0.125, 0.05],
    ])
    edges = [(0, 1), (0, 2), (0, 3), (0, 4)]
    curvatures = [-2.0] * 4

    def loss(iterations):
        mesh = Mesh(SIDES, SIDES, SCALE)
        z = hessian_design.solve_heights(
            mesh, vertices, edges, curvatures, EPSILON,
            iterations=iterations
        )
        computer = TorchComputer(
            mesh, vertices, edges, curvatures, EPSILON, 1.0, 0.0
        )
        return computer.forward(
            hessian_design.calibrate_scale(computer.forward, z) * z
        )

    assert loss(5) < loss(0)


def test_missing_curvature_is_skipped():
    vertices = np.array([[0.08, 0.125], [0.17, 0.125], [0.17, 0.2]])
    _, means = construct(vertices, [(0, 1), (1, 2)], [-1.0, None])
    assert means[1] is None
    assert means[0] < 0
