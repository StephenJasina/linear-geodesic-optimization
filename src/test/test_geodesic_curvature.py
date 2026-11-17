"""Checks for the geodesic curvature of lifted curves."""

import numpy as np
import pytest

from linear_geodesic_optimization import geodesic_curvature as gc
from linear_geodesic_optimization.mesh.rectangle import Mesh


def surface(sides, function):
    mesh = Mesh(sides, sides, 1.)
    xy = mesh.get_coordinates()[:, :2]
    z = function(xy[:, 0], xy[:, 1])
    return mesh, gc.derivative_fields(mesh, z)


def test_tilted_plane_segments_are_geodesics():
    mesh, fields = surface(41, lambda x, y: 0.7 * x - 0.3 * y)
    result = gc.route_curvature(
        mesh, fields, np.array([[0.2, 0.2], [0.8, 0.3], [0.5, 0.8]]), 0.01
    )
    assert result['smooth_turning'] < 1e-9
    # Corners on a plane turn by the angle between the lifted tangents
    gradient = np.array([0.7, -0.3])
    lifted = [np.append(t, gradient @ t) for t in (
        np.array([0.6, 0.1]), np.array([-0.3, 0.5])
    )]
    expected = np.arccos(
        lifted[0] @ lifted[1]
        / np.linalg.norm(lifted[0]) / np.linalg.norm(lifted[1])
    )
    assert np.isclose(result['corner_turning'], expected)


def test_flat_spline_turns_with_its_planar_curvature():
    mesh, fields = surface(21, lambda x, y: 0. * x)
    result = gc.route_curvature(
        mesh, fields, np.array([[0.2, 0.2], [0.5, 0.7], [0.8, 0.3]]),
        0.005, curve='spline',
    )
    assert np.isclose(result['turning'], result['planar_turning'])
    assert np.allclose(
        result['kappa_g'], result['samples'].planar_curvatures
    )


def test_lines_on_a_sphere_lift_to_circles():
    radius = 1.
    mesh, fields = surface(
        201, lambda x, y: np.sqrt(radius**2 - (x - 0.5)**2 - (y - 0.5)**2)
    )
    for offset in (0., 0.1, 0.25):
        waypoints = np.array([[0.2, 0.5 + offset], [0.8, 0.5 + offset]])
        result = gc.route_curvature(mesh, fields, waypoints, 0.01)
        # A vertical plane at distance `offset` from the center cuts a
        # small circle with geodesic curvature d / (R r), turning away
        # from the great circle through the center (here, to the left)
        expected = offset / (radius * np.sqrt(radius**2 - offset**2))
        assert np.allclose(result['kappa_g'], expected, atol=1e-4)


def test_closed_route_satisfies_gauss_bonnet():
    def bump(x, y):
        return 0.2 * np.exp(-((x - 0.45)**2 + (y - 0.5)**2) / 0.02)

    mesh, fields = surface(301, bump)
    triangle = np.array([[0.2, 0.25], [0.8, 0.35], [0.4, 0.8]])
    closed = np.vstack((triangle, triangle[:1]))
    samples = gc.polyline_samples(closed, 0.001)
    derivatives = gc.interpolate(mesh, fields, samples.positions)
    smooth = (
        gc.geodesic_curvature(
            derivatives, samples.tangents, samples.planar_curvatures
        )
        * samples.planar_lengths
        * gc.surface_speed(derivatives, samples.tangents)
    ).sum()
    directions = np.diff(closed, axis=0)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    corners = gc.turning_angle(
        gc.interpolate(mesh, fields, triangle),
        np.roll(directions, 1, axis=0), directions,
    ).sum()

    # Integrate K dA = (h_xx h_yy - h_xy^2) / (1 + |grad h|^2)^(3/2)
    # dx dy over the triangle with a fine midpoint rule
    n = 2000
    x, y = np.meshgrid((np.arange(n) + 0.5) / n, (np.arange(n) + 0.5) / n)
    h = 1e-5
    f = bump
    h_x = (f(x + h, y) - f(x - h, y)) / (2 * h)
    h_y = (f(x, y + h) - f(x, y - h)) / (2 * h)
    h_xx = (f(x + h, y) - 2 * f(x, y) + f(x - h, y)) / h**2
    h_yy = (f(x, y + h) - 2 * f(x, y) + f(x, y - h)) / h**2
    h_xy = (f(x + h, y + h) - f(x + h, y - h)
            - f(x - h, y + h) + f(x - h, y - h)) / (4 * h**2)
    density = (h_xx * h_yy - h_xy**2) / (1 + h_x**2 + h_y**2)**1.5
    a, b, c = triangle

    def side(p, q):
        return (q[0] - p[0]) * (y - p[1]) - (q[1] - p[1]) * (x - p[0])

    inside = (side(a, b) >= 0) & (side(b, c) >= 0) & (side(c, a) >= 0)
    total_curvature = density[inside].sum() / n**2

    assert np.isclose(smooth + corners + total_curvature, 2 * np.pi,
                      atol=2e-3)


def bending_setup():
    mesh = Mesh(41, 41, 1.)
    xy = mesh.get_coordinates()[:, :2]
    z = 0.3 * np.sin(3 * xy[:, 0]) * np.cos(2 * xy[:, 1]) \
        + 0.2 * xy[:, 0]**2 * xy[:, 1]
    curves = [
        np.array([[0.2, 0.2], [0.8, 0.35]]),
        np.array([[0.3, 0.8], [0.7, 0.2]]),
        np.array([[0.2, 0.5], [0.5, 0.75], [0.8, 0.6], [0.7, 0.3]]),
    ]
    return mesh, z, curves, np.array([1., 3., 2.])


@pytest.mark.parametrize('curve', ['polyline', 'spline'])
def test_curve_bending_matches_route_bending(curve):
    mesh, z, curves, weights = bending_setup()
    if curve == 'polyline':
        # Polyline corners aren't counted, so use only straight segments
        curves, weights = curves[:2], weights[:2]
    spacing = 0.25 / 40
    bending = gc.CurveBending(mesh, curves, weights, spacing, curve)
    loss, _ = bending.evaluate(z)
    fields = gc.derivative_fields(mesh, z)
    expected = sum(
        weight / weights.sum() * gc.route_curvature(
            mesh, fields, waypoints, spacing, curve=curve
        )['bending']
        for waypoints, weight in zip(curves, weights)
    )
    assert np.isclose(loss, expected, rtol=1e-12)


@pytest.mark.parametrize('curve', ['polyline', 'spline'])
def test_curve_bending_gradient_matches_directional_difference(curve):
    mesh, z, curves, weights = bending_setup()
    bending = gc.CurveBending(mesh, curves, weights, 0.25 / 40, curve)
    direction = np.random.default_rng(3).normal(size=z.shape)
    step = 1e-6
    numerical = (
        bending.evaluate(z + step * direction)[0]
        - bending.evaluate(z - step * direction)[0]
    ) / (2 * step)
    assert np.isclose(bending.evaluate(z)[1] @ direction, numerical,
                      rtol=1e-5)


def test_route_curves_merge_reversals_and_accumulate_traffic():
    network = ({'labels': ['A', 'B', 'C']}, {}, {})
    routes = [['A', 'B', 'C'], ['C', 'B'], ['C', 'B', 'A'], ['A', 'A']]
    traffic = [1., 2., 4., 5.]
    curves, weights = gc.route_curves(network, routes, traffic, 'route')
    assert dict(zip(curves, weights)) == {(0, 1, 2): 5., (1, 2): 2.}
    curves, weights = gc.route_curves(network, routes, traffic, 'link')
    assert dict(zip(curves, weights)) == {(0, 1): 5., (1, 2): 7.}


@pytest.mark.parametrize('curve', ['polyline', 'spline'])
def test_geodesic_loss_matches_between_backends(curve):
    torch_optimization = pytest.importorskip(
        'linear_geodesic_optimization.optimization.torch_optimization'
    )
    from linear_geodesic_optimization.optimization.optimization import \
        Computer as LegacyComputer
    sides = 12
    mesh = Mesh(sides, sides, 0.25)
    vertices = np.array([[0.05, 0.05], [0.2, 0.06], [0.12, 0.2]])
    edges = [(0, 1), (1, 2), (2, 0)]
    bending = gc.CurveBending(
        mesh, [vertices[[0, 1]], vertices[[0, 1, 2]], vertices[[2, 0]]],
        np.array([1., 2., 0.5]), 0.25 * 0.25 / (sides - 1), curve,
    )
    arguments = (
        mesh, vertices, edges, [0.1, -0.2, 0.05],
        1.01 * 2**0.5 * 0.25 / sides, 1.0, 0.005,
    )
    keywords = {'lambda_geodesic': 0.7, 'geodesic_bending': bending}
    legacy = LegacyComputer(*arguments, **keywords)
    fast = torch_optimization.Computer(*arguments, **keywords)
    z = 0.01 * np.sin(np.arange(sides * sides))
    mesh.set_parameters(z)
    expected_loss = legacy.forward()
    expected_gradient = legacy.reverse()
    assert legacy.geodesic_loss.loss > 0
    np.testing.assert_allclose(fast.forward(z), expected_loss, rtol=1e-9)
    np.testing.assert_allclose(
        fast.geodesic_loss.loss, legacy.geodesic_loss.loss, rtol=1e-9
    )
    np.testing.assert_allclose(
        fast.reverse(z), expected_gradient, rtol=1e-7, atol=1e-7
    )
