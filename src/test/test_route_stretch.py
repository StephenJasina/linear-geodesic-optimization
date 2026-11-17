"""Checks for the route stretch loss."""

import numpy as np
import pytest

pytest.importorskip('potpourri3d')

from linear_geodesic_optimization.mesh.rectangle import Mesh
from linear_geodesic_optimization.route_stretch import RouteStretch


def bump_setup():
    mesh = Mesh(41, 41, 1.)
    xy = mesh.get_coordinates()[:, :2]
    z = 0.15 * np.exp(-((xy[:, 0] - 0.5)**2 + (xy[:, 1] - 0.4)**2) / 0.02)
    routes = [
        np.array([[0.2, 0.3], [0.5, 0.7], [0.8, 0.3]]),
        np.array([[0.2, 0.5], [0.8, 0.5]]),
    ]
    stretch = RouteStretch(mesh, routes, np.array([1., 2.]), 0.25 / 40)
    mesh.set_parameters(z)
    stretch.refresh(z)
    return stretch, z


def test_flat_straight_route_is_not_stretched():
    mesh = Mesh(21, 21, 1.)
    stretch = RouteStretch(
        mesh, [np.array([[0.2, 0.3], [0.8, 0.6]])], np.array([1.]), 0.01
    )
    z = np.zeros(21 * 21)
    stretch.refresh(z)
    assert np.isclose(stretch.stretches(z)[0], 1., atol=1e-6)
    assert stretch.evaluate(z)[0] < 1e-10


def test_routes_over_a_bump_are_longer_than_geodesics():
    stretch, z = bump_setup()
    values = stretch.stretches(z)
    # The detour route is much longer, and the straight route over the
    # bump is longer than the geodesic around it
    assert values[0] > 1.3
    assert 1. < values[1] < 1.1


def test_gradient_matches_directional_difference():
    stretch, z = bump_setup()
    direction = np.random.default_rng(1).normal(size=z.shape)
    step = 1e-6
    numerical = (
        stretch.evaluate(z + step * direction)[0]
        - stretch.evaluate(z - step * direction)[0]
    ) / (2 * step)
    assert np.isclose(stretch.evaluate(z)[1] @ direction, numerical, rtol=1e-5)


def test_gradient_with_smoothing_matches_directional_difference():
    mesh = Mesh(41, 41, 1.)
    xy = mesh.get_coordinates()[:, :2]
    z = 0.15 * np.exp(-((xy[:, 0] - 0.5)**2 + (xy[:, 1] - 0.4)**2) / 0.02)
    stretch = RouteStretch(
        mesh, [np.array([[0.2, 0.3], [0.5, 0.7], [0.8, 0.3]])],
        np.array([1.]), 0.25 / 40, smoothing=0.05,
    )
    stretch.refresh(z)
    direction = np.random.default_rng(2).normal(size=z.shape)
    step = 1e-6
    numerical = (
        stretch.evaluate(z + step * direction)[0]
        - stretch.evaluate(z - step * direction)[0]
    ) / (2 * step)
    assert stretch.evaluate(z)[0] > 0
    assert np.isclose(stretch.evaluate(z)[1] @ direction, numerical, rtol=1e-5)
