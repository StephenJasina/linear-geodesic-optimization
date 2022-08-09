import numpy as np

def wolfe(x, d, f, g, c_1=1e-4, c_2=0.9, max_iterations=100, epsilon=1e-6):
    '''
    Compute a step size satisfying the (weak) Wolfe conditions. We assume that
    f is a scalar-valued function, g is its f's gradient, and d is a descent
    direction.

    If we define phi(alpha) = f(x + alpha * d), then we have
    phi'(alpha) = g(x + alpha * d) @ d. The Wolfe conditions are then
      * phi(alpha) <= phi(0) + c_1 * phi'(0) * alpha
      * phi'(alpha) >= c_2 * phi'(0)
    '''

    alpha_l = 0.
    alpha_r = np.inf
    alpha = min(1., 100. / (1. + np.linalg.norm(d)))

    d = d / np.linalg.norm(d)

    phi_0 = f(x)
    dphi_0 = g(x) @ d

    for _ in range(max_iterations):
        phi_alpha = f(x + alpha * d)
        if phi_alpha <= phi_0 + c_1 * alpha * dphi_0:
            dphi_alpha = g(x + alpha * d) @ d
            if dphi_alpha >= c_2 * dphi_0:
                return alpha
            alpha_l = alpha
        else:
            alpha_r = alpha
        if np.isposinf(alpha_r):
            alpha = 2 * alpha_l
        else:
            alpha = (alpha_l + alpha_r) / 2
            if alpha_r - alpha_l < epsilon:
                return None
