import numpy as np

from linear_geodesic_optimization.optimization import linear_regression

linear_regression_forward = linear_regression.Forward()
linear_regression_reverse = linear_regression.Reverse(linear_regression_forward)

rng = np.random.default_rng()

delta = 1e-5
phi_0 = rng.random(16)
dif_phi_0 = rng.random(16)
t = (phi_0 * 10 + 4) + 0.1 * rng.random(16)

linear_regression_forward.calc(phi_0, t)
lse_0 = linear_regression_forward.lse

for l in range(phi_0.shape[0]):
    linear_regression_reverse.calc(phi_0, t, dif_phi_0, l)
    dif_lse_0 = linear_regression_reverse.dif_lse

    phi_delta = np.copy(phi_0)
    phi_delta += delta * dif_phi_0
    linear_regression_forward.calc(phi_delta, t)
    lse_delta = linear_regression_forward.lse
    print(((lse_delta - lse_0) / delta) / dif_lse_0)
