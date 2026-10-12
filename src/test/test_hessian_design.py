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
    """Return (heights, mean kappa_G per edge)."""
    mesh = Mesh(SIDES, SIDES, SCALE)
    z = hessian_design.solve_heights(
        mesh, vertices, edges, curvatures, EPSILON, **kwargs
    )
    computer = TorchComputer(
        mesh, vertices, edges, curvatures, EPSILON, 1.0, 0.0
    )
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
        return computer.forward(z)

    assert loss(5) < loss(0)


def test_missing_curvature_is_skipped():
    vertices = np.array([[0.08, 0.125], [0.17, 0.125], [0.17, 0.2]])
    _, means = construct(vertices, [(0, 1), (1, 2)], [-1.0, None])
    assert means[1] is None
    assert means[0] < 0


def test_gradient_operators_are_exact_on_quadratics():
    sides = 12
    dx, dy = hessian_design.gradient_operators(sides, SCALE)
    interior = hessian_design.hessian_operators(sides, SCALE)[3]
    xy = Mesh(sides, sides, SCALE).get_coordinates()[:, :2]
    x, y = xy[:, 0], xy[:, 1]
    f = 0.3 * x**2 + 0.7 * x * y - 1.1 * y**2
    np.testing.assert_allclose(dx @ f, (0.6 * x + 0.7 * y)[interior], atol=1e-10)
    np.testing.assert_allclose(dy @ f, (0.7 * x - 2.2 * y)[interior], atol=1e-10)


def test_slope_correction_matches_steep_curvature():
    # On a large mesh the uncorrected design is steep enough that the
    # small-slope approximation K = det(Hess f) undershoots the target
    sides, scale = 40, 2.
    epsilon = 1.01 * 2**0.5 * scale / sides
    vertices = np.array([[0.64, 1.], [1.36, 1.]])
    curvature = -2.

    def mean_kappa_G(slope_correction):
        mesh = Mesh(sides, sides, scale)
        z = hessian_design.solve_heights(
            mesh, vertices, [(0, 1)], [curvature], epsilon,
            iterations=10, slope_correction=slope_correction
        )
        computer = TorchComputer(
            mesh, vertices, [(0, 1)], [curvature], epsilon, 1.0, 0.0
        )
        computer.forward(z)
        (residual,) = hessian_design.edge_residuals(
            computer.kappa_G, mesh.get_coordinates()[:, :2],
            hessian_design.hessian_operators(sides, scale)[3],
            vertices, [(0, 1)], [curvature], epsilon
        )
        return residual['mean_kappa_G']

    uncorrected = mean_kappa_G(False)
    corrected = mean_kappa_G(True)
    assert abs(corrected - curvature) < 0.5 * abs(uncorrected - curvature)


def tensor_smooth_loss(mesh, z):
    """Return the tensor smoothness loss of heights z."""
    computer = TorchComputer(
        mesh, np.array([[0., 0.], [1., 1.]]), [(0, 1)], [None],
        1e-9, 1.0, 1.0, smooth_loss_type='tensor'
    )
    computer.forward(z)
    return computer.smooth_loss.loss


def bump(mesh, amplitude):
    """Return a sin(pi x / L) sin(pi y / L), which is 0 on the boundary."""
    xy = mesh.get_coordinates()[:, :2] / mesh.get_scale()
    return amplitude * np.sin(np.pi * xy[:, 0]) * np.sin(np.pi * xy[:, 1])


@pytest.mark.parametrize('amplitude', [0., 0.02, 0.1])
def test_smoothness_operator_equals_tensor_loss_at_linearization(amplitude):
    # Freezing every coefficient at z and evaluating at z gives back the
    # true loss, steep or not
    sides, scale = 16, 0.25
    mesh = Mesh(sides, sides, scale)
    z = bump(mesh, amplitude) + 0.01 * bump(Mesh(sides, sides, scale / 3), 1.)
    z[~hessian_design.hessian_operators(sides, scale)[3]] = 0.
    q = hessian_design.smoothness_operator(
        mesh, z, clamp_negative_weights=False
    )
    interior = hessian_design.hessian_operators(sides, scale)[3]
    np.testing.assert_allclose(
        z[interior] @ q @ z[interior], tensor_smooth_loss(mesh, z), rtol=1e-9
    )


def test_flat_smoothness_operator_converges_to_third_derivative_energy():
    # For a sin(pi x / L) sin(pi y / L), A_total times the integral of
    # ||grad Hess f||^2 is 2 a^2 pi^6 / L^2
    errors = []
    scale = 0.5
    for sides in (20, 40, 80):
        mesh = Mesh(sides, sides, scale)
        interior = hessian_design.hessian_operators(sides, scale)[3]
        f = bump(mesh, 1.)[interior]
        q = hessian_design.smoothness_operator(mesh)
        errors.append(abs(f @ q @ f / (2 * np.pi**6 / scale**2) - 1.))
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 0.1


def test_tensor_loss_ignores_uniform_curvature_for_small_slopes():
    # A quadratic has a constant Hessian, so its loss vanishes to leading
    # order, unlike a surface of varying curvature (the thin plate
    # regularizer would penalize both)
    sides, scale = 16, 0.25
    mesh = Mesh(sides, sides, scale)
    xy = mesh.get_coordinates()[:, :2] - scale / 2
    quadratic = 0.3 * xy[:, 0]**2 + 0.7 * xy[:, 0] * xy[:, 1] \
        - 1.1 * xy[:, 1]**2
    eps = 1e-3
    assert tensor_smooth_loss(mesh, eps * quadratic) \
        < 1e-6 * tensor_smooth_loss(mesh, eps * bump(mesh, 0.01))


def test_tensor_loss_reduces_to_third_derivative_energy_for_small_slopes():
    sides, scale = 20, 0.5
    mesh = Mesh(sides, sides, scale)
    interior = hessian_design.hessian_operators(sides, scale)[3]
    f = bump(mesh, 1.)
    s3 = f[interior] @ hessian_design.smoothness_operator(mesh) @ f[interior]
    ratios = [tensor_smooth_loss(mesh, eps * f) / eps**2 / s3
              for eps in (1e-1, 1e-2, 1e-3)]
    assert abs(ratios[2] - 1.) < 1e-4
    assert abs(ratios[2] - 1.) < abs(ratios[0] - 1.)


@pytest.mark.parametrize('surface', ['sphere', 'cylinder'])
def test_tensor_loss_vanishes_on_constant_shape_operator(surface):
    # Spheres and cylinders have constant shape operators up to
    # transport, so their losses shrink under refinement, unlike a
    # surface of varying curvature at a similar scale
    radius = 0.6

    def losses(sides):
        mesh = Mesh(sides, sides, 0.5)
        x, y = (mesh.get_coordinates()[:, :2] - 0.25).T
        r2 = x**2 + (y**2 if surface == 'sphere' else 0.)
        z = np.sqrt(radius**2 - r2)
        return tensor_smooth_loss(mesh, z), \
            tensor_smooth_loss(mesh, 0.2 * bump(mesh, 1.))

    coarse, varying = losses(20)
    fine, _ = losses(40)
    assert coarse < 1e-2 * varying
    assert fine < coarse / 2


def test_smoothness_reduces_tensor_loss():
    vertices = np.array([[0.08, 0.08], [0.17, 0.08], [0.17, 0.17]])
    edges = [(0, 1), (1, 2), (0, 2)]
    curvatures = [-1.0, 0.5, -0.5]

    def losses(smoothness):
        mesh = Mesh(SIDES, SIDES, SCALE)
        z = hessian_design.solve_heights(
            mesh, vertices, edges, curvatures, EPSILON,
            iterations=3, smoothness=smoothness
        )
        return tensor_smooth_loss(mesh, z)

    assert losses(1e-4) < 0.5 * losses(0.)
