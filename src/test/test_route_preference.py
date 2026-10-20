"""A route preference should change the free shortest path, not just its drawing."""

import numpy as np
import pytest
from scipy.optimize import minimize
import torch

from linear_geodesic_optimization.mesh.rectangle import Mesh
from linear_geodesic_optimization.optimization.route_preference import (
    RouteConstraint,
    RoutePreference,
)


def test_route_loss_gradient_and_free_path_fit():
    mesh = Mesh(15, 15, 0.25)
    waypoints = np.array([
        [0.03, 0.125], [0.125, 0.22], [0.22, 0.125]
    ])
    preference = RoutePreference(
        mesh, waypoints, [RouteConstraint((0, 1, 2))], 0.09, 0.02
    )
    start = 0.003 * np.sin(np.arange(225) * 1.731)
    before = preference.metrics(start)[0]
    assert before['fraction_free_path_in_corridor'] < 0.5
    assert before['relative_length_gap'] > 0.1
    guided = preference.guided_path(start)
    assert guided['free_corridor_fraction'] < 0.5
    assert guided['guided_corridor_fraction'] == 1.0
    assert guided['strength'] > 0

    direction = np.random.default_rng(23).normal(size=start.size)
    direction /= np.linalg.norm(direction)
    heights = torch.tensor(start, requires_grad=True)
    loss = preference.loss(heights)
    gradient, = torch.autograd.grad(loss, heights)
    step = 1e-7
    numerical = (
        float(preference.loss(torch.tensor(start + step * direction)))
        - float(preference.loss(torch.tensor(start - step * direction)))
    ) / (2 * step)
    np.testing.assert_allclose(
        float(gradient @ torch.tensor(direction)), numerical,
        rtol=1e-4, atol=1e-5,
    )

    def objective(values):
        variable = torch.tensor(values, requires_grad=True)
        value = preference.loss(variable) + 0.0001 * (variable**2).sum()
        derivative, = torch.autograd.grad(value, variable)
        return float(value.detach()), derivative.detach().numpy()

    optimized = minimize(
        objective, start, jac=True, method='L-BFGS-B',
        bounds=[(-0.2, 0.2)] * start.size,
        options={'maxiter': 60},
    )
    after = preference.metrics(optimized.x)[0]
    assert after['fraction_free_path_in_corridor'] > 0.9
    assert after['relative_length_gap'] < before['relative_length_gap']


def test_disconnected_corridor_is_reported():
    mesh = Mesh(10, 10, 0.25)
    points = np.array([[0.025, 0.025], [0.125, 0.225], [0.225, 0.025]])
    with pytest.raises(ValueError, match='increase route_corridor_width'):
        preference = RoutePreference(
            mesh, points, [RouteConstraint((0, 1, 2))], 0.001, 0.0
        )
        preference.metrics(np.zeros(100))
