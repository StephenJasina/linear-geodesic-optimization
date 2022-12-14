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

    def calc(self, phi, t):
        E = phi.shape[0]

        self._phi = np.copy(phi)
        self._t = np.copy(t)

        self.d_tilde = phi - np.sum(phi) / E
        self.d = self.d_tilde / (self.d_tilde @ self.d_tilde / E)**0.5
        self.residuals = t - np.sum(t) / E - (self.d @ t / E) * self.d
        # TODO: Fix the scaling here
        self.lse = self.residuals @ self.residuals / (t.shape[0] * t @ t)

    def get_beta(self, phi, t):
        # Note that the following disagrees with the notation in the writeup
        # In particular, beta here is the coefficients when relating phi to t
        # (as opposed to relating d to t).
        E = phi.shape[0]
        denominator = E * (phi @ phi) - np.sum(phi)**2
        return (
            (phi @ phi * np.sum(t) - np.sum(phi) * (phi @ t)) / denominator,
            (E * (phi @ t) - np.sum(phi) * np.sum(t)) / denominator,
        )

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

    def calc(self, phi, t, dif_phi):
        E = phi.shape[0]

        self._phi = np.copy(phi)
        self._t = np.copy(t)
        self._dif_phi = dif_phi

        self._linear_regression_forward.calc(self._phi, self._t)
        self._d_tilde = self._linear_regression_forward.d_tilde
        self._d = self._linear_regression_forward.d
        self._residuals = self._linear_regression_forward.residuals
        self._lse = self._linear_regression_forward.lse

        self.dif_d_tilde = self._dif_phi - np.sum(self._dif_phi) / E
        self.dif_d = (self.dif_d_tilde
                      - (self._d @ self.dif_d_tilde)
                        * self._d / E) / (linalg.norm(self._d_tilde) / E**0.5)
        self.dif_residuals = -(self._d @ self._t * self.dif_d
                               + self.dif_d @ self._t * self._d) / E
        # TODO: Fix the scaling here
        self.dif_lse = 2 * self._residuals @ self.dif_residuals / (t.shape[0] * t @ t)
