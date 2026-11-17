"""Checks for the greedy geodesic nudge."""

import numpy as np
import pytest

pytest.importorskip('potpourri3d')

from linear_geodesic_optimization import geodesic_nudge as gn
from linear_geodesic_optimization.mesh.rectangle import Mesh
from linear_geodesic_optimization.route_stretch import RouteStretch


def lateral_offsets(points, normals, curve):
    """Signed distance along each normal from `points` to the polyline `curve`."""
    offsets = np.full(len(points), np.nan)
    for index, (point, normal) in enumerate(zip(points, normals)):
        for a, b in zip(curve[:-1], curve[1:]):
            matrix = np.column_stack((normal, a - b))
            if abs(np.linalg.det(matrix)) < 1e-14:
                continue
            t, s = np.linalg.solve(matrix, a - point)
            if -1e-9 <= s <= 1 + 1e-9 and (np.isnan(offsets[index]) or abs(t) < abs(offsets[index])):
                offsets[index] = t
    return offsets


def bump_surface(sides):
    mesh = Mesh(sides, sides, 1.)
    xy = mesh.get_coordinates()[:, :2]
    z = 0.15 * np.exp(-((xy[:, 0] - 0.5)**2 + (xy[:, 1] - 0.45)**2) / 0.02)
    return mesh, xy, z


def test_predicted_shift_matches_retraced_geodesic():
    sides = 81
    mesh, xy, z = bump_surface(sides)
    route = np.array([[0.2, 0.5], [0.8, 0.5]])
    spacing = 0.5 / (sides - 1)
    tracer = RouteStretch(mesh, [route], np.array([1.]), spacing)
    tracer.refresh(z)
    linearization = gn.RouteLinearization(
        mesh, z, tracer.competitors[0], route, 1., spacing, 1 / (sides - 1)
    )
    assert linearization.valid
    dz = np.exp(-((xy[:, 0] - 0.45)**2 + (xy[:, 1] - 0.6)**2) / 0.01)
    epsilon = 0.01
    predicted = epsilon * linearization.displacement(dz)
    tracer.refresh(z + epsilon * dz)
    actual = lateral_offsets(
        linearization.points[1:-1], linearization.normals[1:-1], tracer.competitors[0]
    )
    known = ~np.isnan(actual)
    assert known.mean() > 0.9
    assert np.corrcoef(predicted[known], actual[known])[0, 1] > 0.99
    assert np.linalg.norm(predicted[known] - actual[known]) \
        < 0.1 * np.linalg.norm(actual[known])


def test_offset_gradient_matches_directional_difference():
    sides = 41
    mesh, xy, z = bump_surface(sides)
    route = np.array([[0.2, 0.5], [0.5, 0.6], [0.8, 0.5]])
    spacing = 0.5 / (sides - 1)
    tracer = RouteStretch(mesh, [route], np.array([1.]), spacing)
    tracer.refresh(z)
    linearization = gn.RouteLinearization(
        mesh, z, tracer.competitors[0], route, 1., spacing, 1 / (sides - 1)
    )
    dz = 0.1 * np.random.default_rng(0).normal(size=z.shape)
    direction = np.random.default_rng(1).normal(size=z.shape)
    step = 1e-6
    numerical = (
        linearization.evaluate(dz + step * direction)[0]
        - linearization.evaluate(dz - step * direction)[0]
    ) / (2 * step)
    assert np.isclose(linearization.evaluate(dz)[1] @ direction, numerical, rtol=1e-5)
    assert np.isclose(linearization.evaluate(np.zeros_like(z))[0], linearization.offset())


class FlatLoss:
    """A stand-in loss computer: the squared distance to the start heights."""

    def __init__(self, z):
        self.z = z
        self.curvature_loss = type('Loss', (), {'loss': 0.})()
        self.smooth_loss = type('Loss', (), {'loss': 0.})()

    def forward(self, z):
        self.curvature_loss.loss = float(np.sum((z - self.z)**2)) + 1.
        return self.curvature_loss.loss

    def reverse(self, z):
        return 2 * (z - self.z)


def test_nudge_moves_geodesic_toward_route():
    sides = 41
    mesh, xy, z = bump_surface(sides)
    # The route crosses the bump; its geodesic goes around it
    route = np.array([[0.2, 0.45], [0.5, 0.45], [0.8, 0.45]])
    nudge = gn.GeodesicNudge(
        mesh, [route], np.array([1.]), FlatLoss(z), lambda_fit=0.01,
        step_smoothing=0.05, inner_maxiter=50,
    )
    z_new, history = nudge.run(z, 3, log=lambda line: None)
    accepted = [entry for entry in history if entry.get('accepted')]
    assert len(accepted) > 1
    assert accepted[-1]['offset'] < 0.5 * history[0]['offset']
    assert accepted[-1]['miss_mean'] < history[0]['miss_mean']
