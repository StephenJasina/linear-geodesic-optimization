import numpy as np
from scipy import linalg

class Forward:
    def __init__(self):
        self._phi = None
        self._t = None

        self.d_tilde = None
        self.d = None
        self.residuals = None
        self.lse = None

        self.beta = None

    def calc(self, phi, t):
        E = phi.shape[0]

        # If phi and t haven't changed, don't do any additional work.
        if (self._phi is not None and self._phi.shape == phi.shape
            and np.allclose(self._phi, phi)
            and self._t is not None and self._t.shape == t.shape
            and np.allclose(self._t, t)
            and self.lse is not None):
            return self.lse

        self._phi = np.copy(phi)
        self._t = np.copy(t)

        self.d_tilde = phi - np.sum(phi) / E
        self.d = self.d_tilde / (self.d_tilde @ self.d_tilde / E)**0.5
        self.residuals = t - np.sum(t) / E - (self.d @ t / E) * self.d
        self.lse = self.residuals @ self.residuals
        self.beta = None

    def get_beta(self, phi, t):
        E = phi.shape[0]

        # If phi and t haven't changed, don't do any additional work.
        if (self._phi is not None and self._phi.shape == phi.shape
            and np.allclose(self._phi, phi)
            and self._t is not None and self._t.shape == t.shape
            and np.allclose(self._t, t)
            and self.beta is not None):
            return self.beta

        self._phi = np.copy(phi)
        self._t = np.copy(t)

        # Note that the following disagrees with the notation in the writeup
        # above. In particular, beta here is the coefficients when relating phi
        # to t (as opposed to relating d to t).
        denominator = E * (phi @ phi) - np.sum(phi)**2
        self.beta = (
            (phi @ phi * np.sum(t) - np.sum(phi) * (phi @ t)) / denominator,
            (E * (phi @ t) - np.sum(phi) * np.sum(t)) / denominator,
        )
        self.d_tilde = None
        self.d = None
        self.residuals = None
        self.lse = None

        return self.beta

class Reverse:
    def __init__(self, linear_regression_forward=None):
        self._phi = None
        self._t = None
        self._dif_phi = None

        self._l = None

        self._linear_regression_forward = linear_regression_forward
        if linear_regression_forward is None:
            self._linear_regression_forward = Forward()

        self._d_tilde = None
        self._d = None
        self._residuals = None
        self._lse = None

        self.dif_d_tilde = None
        self.dif_d = None
        self.dif_residuals = None
        self.dif_lse = None

    def calc(self, phi, t, dif_phi, l):
        E = phi.shape[0]

        pair_changed = False
        if (self._phi is None or self._phi.shape != phi.shape
            or not np.allclose(self._phi, phi)
            or self._t is None or self._t.shape != t.shape
            or not np.allclose(self._t, t) or self._l != l
            or self._dif_phi is None
            or not np.allclose(self._dif_phi, dif_phi)):
            pair_changed = True
            self._phi = np.copy(phi)
            self._t = np.copy(t)
            self._dif_phi = dif_phi

            self._l = l

            self._linear_regression_forward.calc(self._phi, self._t)
            self._d_tilde = self._linear_regression_forward.d_tilde
            self._d = self._linear_regression_forward.d
            self._residuals = self._linear_regression_forward.residuals
            self._lse = self._linear_regression_forward.lse

        if not pair_changed:
            return self.dif_lse

        self.dif_d_tilde = self._dif_phi - np.sum(self._dif_phi) / E
        self.dif_d = (self.dif_d_tilde - (self._d @ self.dif_d_tilde) * self._d) / linalg.norm(self._d_tilde)
        self.dif_residuals = (-self._d @ self._t * self.dif_d - self.dif_d @ self._t * self._d) / E
        self.dif_lse = 2 * self._residuals @ self.dif_residuals
