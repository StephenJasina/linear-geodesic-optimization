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

    def calc_lse(self, phi, t):
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
        return self.lse

    def calc_beta(self, phi, t):
        E = phi.shape[0]

        # If phi and t haven't changed, don't do any additional work.
        if (self._phi is not None and self._phi.shape == phi.shape
            and np.allclose(self._phi, phi)
            and self._t is not None and self._t.shape == t.shape
            and np.allclose(self._t, t)
            and self.beta is not None):
            return self.beta

        # Note that the following disagrees with the notation in the writeup
        # above. In particular, beta here is the coefficients when relating phi
        # to t (as opposed to relating d to t).
        denominator = E * (phi @ phi) - np.sum(phi)**2
        self.beta = (
            (phi @ phi * np.sum(t) - np.sum(phi) * (phi @ t)) / denominator,
            (E * (phi @ t) - np.sum(phi) * np.sum(t)) / denominator,
        )
        return self.beta

class Reverse:
    def __init__(self, linear_regression_forward=None):
        self._phi = None
        self._t = None
        self._dif_phi = None

        self._ls = None

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

    def calc_dif_lse(self, phi, t, dif_phi, ls=None):
        E = phi.shape[0]
        if self._ls is None:
            self._ls = range(E)
        if ls is None:
            ls = range(E)

        pair_changed = False
        if (self._phi is None or self._phi.shape != phi.shape
            or not np.allclose(self._phi, phi)
            or self._t is None or self._t.shape != t.shape
            or not np.allclose(self._t, t) or list(self._ls) != list(ls)
            or self._dif_phi is None
            or not np.all([l in self._dif_phi for l in self._ls])
            or not np.all([l in dif_phi for l in self._ls])
            or not np.all([np.allclose(self._dif_phi[l], dif_phi[l])
                           for l in self._ls])):
            pair_changed = True
            self._phi = np.copy(phi)
            self._t = np.copy(t)
            self._dif_phi = dif_phi

            self._ls = ls

            self._linear_regression_forward.calc_lse(self._phi, self._t)
            self._d_tilde = self._linear_regression_forward.d_tilde
            self._d = self._linear_regression_forward.d
            self._residuals = self._linear_regression_forward.residuals
            self._lse = self._linear_regression_forward.lse

        if not pair_changed:
            return self.dif_lse

        self.dif_d_tilde = {l: self._dif_phi[l] - np.sum(self._dif_phi[l]) / E for l in self._ls}
        self.dif_d = {l: (self.dif_d_tilde[l] - (self._d @ self.dif_d_tilde[l]) * self._d) / linalg.norm(self._d_tilde) for l in self._ls}
        self.dif_residuals = {l: (-self._d @ self._t * self.dif_d[l] - self.dif_d[l] @ self._t * self._d) / E for l in self._ls}
        self.dif_lse = {l: 2 * self._residuals @ self.dif_residuals[l] for l in self._ls}
        return self.dif_lse