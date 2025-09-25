'''
Implementation of some standard optimization-related methods.
'''

import itertools

import numpy as np


def wolfe(f, g, x, d, c_1=1e-4, c_2=0.9, max_iterations=100, epsilon=1e-6):
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

def steepest_descent(f, g, x, max_iterations, diagnostics=None):
    '''
    Implementation of the steepest descent (a.k.a. gradient descent) algorithm,
    where the step size is chosen to satisfy the Wolfe conditions.
    '''

    for k in itertools.count():
        if k >= max_iterations:
            break

        if diagnostics is not None:
            diagnostics(k)

        d = -g(x)
        alpha = wolfe(x, d, f, g)

        if alpha is None:
            # We are pretty much stuck, so end here
            break

        x = x + alpha * d

    if diagnostics is not None:
        diagnostics()

def lbfgs(f, g, x, max_iterations, diagnostics=None, m=5):
    '''
    Implementation of the Limited-memory BFGS algorithm, where the step size is
    chosen to satisfy the Wolfe conditions.
    '''

    H_0 = 1 # H is a scalar multiple of the identity
    ss = []
    ys = []

    for k in itertools.count():
        if k >= max_iterations:
            break

        if diagnostics is not None:
            diagnostics(k)

        # Two loop recursion
        g_k = g(x)
        q = g_k
        etas = []
        for s, y in zip(reversed(ss), reversed(ys)):
            eta = s @ q / (s @ y)
            etas.append(eta)
            q = q - eta * y
        r = H_0 * q
        for s, y, eta in zip(ss, ys, reversed(etas)):
            beta = r @ y / (s @ y)
            r = r + (eta - beta) * s

        d = -r
        alpha = wolfe(x, d, f, g)
        if alpha is None:
            # If d is not a descent direction, use steepest descent instead
            d = -g_k
            alpha = wolfe(x, d, f, g)
            if alpha is None:
                # We are pretty much stuck, so end here
                break
            x = x + alpha * d

            # Also reset the Hessian estimation
            H_0 = 1
            ss = []
            ys = []
        else:
            new_x = x + alpha * d
            new_g_k = g(new_x)

            if len(ss) >= m:
                del ss[0]
                del ys[0]

            s = new_x - x
            y = new_g_k - g_k

            ss.append(s)
            ys.append(y)
            x = new_x
            H_0 = s @ y / (y @ y)

    if diagnostics is not None:
        diagnostics()
