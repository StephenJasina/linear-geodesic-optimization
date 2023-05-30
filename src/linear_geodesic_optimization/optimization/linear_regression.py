"""Module containing utilities for linear regression."""

import typing

import numpy as np
import numpy.typing as npt
from scipy import linalg


class Computer:
    """Implementation of linear regression."""

    def __init__(
        self,
        phi: npt.NDArray[np.float64],
        t: npt.NDArray[np.float64],
        dif_phi: typing.List[typing.Dict[int, np.float64]]
    ):
        """Initialize the computer."""
        # Forward variables
        self._d: npt.NDArray[np.float64] = np.zeros(phi.shape)
        self._d_tilde: npt.NDArray[np.float64] = np.zeros(phi.shape)
        self.phi: npt.NDArray[np.float64] = phi
        self.t: npt.NDArray[np.float64] = t
        self.beta: typing.Tuple[np.float64, np.float64] \
            = (np.float64(0.), np.float64(1.))
        self.residuals: npt.NDArray[np.float64] = np.zeros(phi.shape)
        self.loss: np.float64

        # Reverse variables
        self._dif_d_tilde: typing.List[typing.Dict[int, np.float64]] = []
        self._dif_d: typing.List[typing.Dict[int, np.float64]] = []
        self.dif_phi: typing.List[typing.Dict[int, np.float64]] = dif_phi
        self.dif_loss: typing.Dict[int, np.float64] = {}

    def forward(self) -> None:
        """
        Compute the forward direction.

        The computed values will be stored in the variables:
        * `Computer.beta`
        * `Computer.residuals`
        * `Computer.loss`
        """
        e = self.phi.shape[0]
        if e == 0:
            self.beta = (np.float64(0.), np.float64(1.))
            self.residuals = np.array([])
            self.loss = np.float64(0.)
            return

        beta_denominator = e * (self.phi @ self.phi) - np.sum(self.phi)**2
        self.beta = (
            (self.phi @ self.phi * np.sum(self.t)
             - np.sum(self.phi) * (self.phi @ self.t)) / beta_denominator,
            (e * (self.phi @ self.t)
             - np.sum(self.phi) * np.sum(self.t)) / beta_denominator
        )
        self._d_tilde = self.phi - np.sum(self.phi) / e
        self._d = self._d_tilde / (self._d_tilde @ self._d_tilde / e)**0.5
        self.residuals = self.t - np.sum(self.t) / e \
            - (self._d @ self.t / e) * self._d
        self.loss = self.residuals @ self.residuals \
            / (e * np.var(self.t, ddof=1))

    def reverse(self) -> None:
        """
        Compute the reverse direction (that is, partials).

        The computed values will be stored in the variables:
        * `Computer.dif_loss`
        """
        self.forward()

        e = self.phi.shape[0]
        if e == 0:
            self.dif_loss = {}
            return

        # TODO: Finish this
        indices = set()
        indices.union(*[element.keys() for element in self.dif_phi])

        self._dif_d_tilde = []
        self._dif_d = []
        for phi, element in self.phi, self.dif_phi:
            dif_d_tilde = {}
            dif_d = {}
            for index, partial in element.items():
                dif_d_tilde[index]


        for index, partial in self.dif_phi:
            self._dif_d_tilde[index] = self._dif_phi - np.sum(self._dif_phi) / e
            self._dif_d[index] = (dif_d_tilde
                        - (self._d @ dif_d_tilde)
                            * self._d / e) / (linalg.norm(self._d_tilde) / e**0.5)
            self.dif_residuals = -(self._d @ self._t * self.dif_d
                                + self.dif_d @ self._t * self._d) / e
            self.dif_lse = 2 * self.residuals @ self.dif_residuals \
                / (e * np.var(self.t, ddof=1))

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

        if E == 0:
            self.dif_d_tilde = np.array([])
            self.dif_d = np.array([])
            self.dif_residuals = np.array([])
            self.dif_lse = np.array([])
            return

        self.dif_d_tilde = self._dif_phi - np.sum(self._dif_phi) / E
        self.dif_d = (self.dif_d_tilde
                      - (self._d @ self.dif_d_tilde)
                        * self._d / E) / (linalg.norm(self._d_tilde) / E**0.5)
        self.dif_residuals = -(self._d @ self._t * self.dif_d
                               + self.dif_d @ self._t * self._d) / E
        self.dif_lse = 2 * self._residuals @ self.dif_residuals \
            / (t.shape[0] * np.var(t, ddof=1))
