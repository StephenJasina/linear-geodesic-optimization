"""Check that the continuous route objective follows the actual surface geodesic."""

import numpy as np
from scipy.optimize import minimize
import torch

from linear_geodesic_optimization.mesh.rectangle import Mesh
from linear_geodesic_optimization.optimization.continuous_route_preference import (
    ContinuousRoutePreference,
)
from linear_geodesic_optimization.optimization.route_preference import RouteConstraint


def test_continuous_route_gradient_and_geodesic_fit():
    mesh = Mesh(15, 15, 0.25)
    points = np.array([[0.03, 0.125], [0.125, 0.22], [0.22, 0.125]])
    preference = ContinuousRoutePreference(
        mesh, points, [RouteConstraint((0, 1, 2))], 0.09, 0.0
    )
    start = 0.003 * np.sin(np.arange(225) * 1.731)
    before = preference.metrics(start)[0]
    assert before['fraction_free_path_in_corridor'] < 0.5

    direction = np.random.default_rng(23).normal(size=start.size)
    direction /= np.linalg.norm(direction)
    heights = torch.tensor(start, requires_grad=True)
    gradient, = torch.autograd.grad(preference.loss(heights), heights)
    step = 1e-7
    numerical = (
        float(preference.loss(torch.tensor(start + step * direction)))
        - float(preference.loss(torch.tensor(start - step * direction)))
    ) / (2 * step)
    np.testing.assert_allclose(float(gradient @ torch.tensor(direction)), numerical,
                               rtol=1e-4, atol=1e-5)

    def objective(values):
        variable = torch.tensor(values, requires_grad=True)
        value = preference.loss(variable) + 0.0001 * (variable**2).sum()
        derivative, = torch.autograd.grad(value, variable)
        return float(value.detach()), derivative.detach().numpy()

    result = minimize(objective, start, jac=True, method='L-BFGS-B',
                      options={'maxiter': 30})
    after = preference.metrics(result.x)[0]
    assert after['fraction_free_path_in_corridor'] > 0.95
    assert after['relative_length_gap'] < before['relative_length_gap']
