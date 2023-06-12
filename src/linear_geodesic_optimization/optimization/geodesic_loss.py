"""Module containing utilities for geodesic loss."""

import typing

import numpy as np
import numpy.typing as npt

from linear_geodesic_optimization.optimization.geodesic \
    import Computer as Geodesic


class Computer:
    """Implementation of geodesic loss."""

    def __init__(self, geodesics: typing.List[Geodesic],
                 t: npt.NDArray[np.float64]):
        """
        Initialize the computer.

        This computer finds scalars such that the objective
          ||(beta_0 + beta_1 * phi) - t||_2
        is minimized, where phi is a vector of geodesic distances.
        """
        self._geodesics = geodesics
        self.t = t
        """The dependent variable."""

        # Forward variables
        self.phi: npt.NDArray[np.float64] = np.array([])
        """The independent variable."""
        self.beta: typing.Tuple[np.float64, np.float64] = (np.float64(0.),
                                                           np.float64(1.))
        """The regression coefficients."""
        self.residuals: npt.NDArray[np.float64] = np.zeros(self.phi.shape)
        """The residuals after doing linear regression."""
        self.loss: np.float64 = np.float64(0.)
        """The sum of the squares of the residuals."""

        # Reverse variables
        self.dif_phi: typing.List[typing.Dict[int, np.float64]] = []
        """
        The partials of the independent variable, indexed by vertices.
        """
        self.dif_loss: typing.Dict[int, np.float64] = {}
        """The partials of the sum of the squares of the residuals,
        indexed by vertices.
        """

    def forward(self) -> None:
        """
        Compute the forward direction.

        The computed values will be stored in the variables:
        * `Computer.beta`
        * `Computer.residuals`
        * `Computer.loss`
        """
        phi_list: typing.List[np.float64] = []
        for geodesic in self._geodesics:
            geodesic.forward()
            phi_list.append(geodesic.distance)
        self.phi = np.array(phi_list)

        n = self.phi.shape[0]

        sum_phi = sum(self.phi)
        sum_t = sum(self.t)
        phi_phi = self.phi @ self.phi
        phi_t = self.phi @ self.t
        nu = (
            sum_t * phi_phi - sum_phi * phi_t,
            n * phi_t - sum_phi * sum_t
        )
        delta = n * phi_phi - sum_phi * sum_phi
        centered_t = self.t - sum_t / n
        n_var_t = centered_t @ centered_t

        if n <= 1:
            self.beta = (np.float64(0.), np.float64(1.))
            if n == 0:
                self.residuals = np.array([])
            else:
                self.residuals = np.array([np.float64(0.)])
            self.loss = np.float64(0.)
            return

        self.beta = (nu[0] / delta, nu[1] / delta)
        self.residuals = (self.beta[0] + self.beta[1] * self.phi) - self.t
        self.loss = self.residuals @ self.residuals / n_var_t

    def reverse(self) -> None:
        """
        Compute the reverse direction (that is, partials).

        The computed values will be stored in the variables:
        * `Computer.dif_loss`
        """
        self.forward()
        self.dif_phi = []
        for geodesic in self._geodesics:
            geodesic.reverse()
            self.dif_phi.append(geodesic.dif_distance)

        n = self.phi.shape[0]
        if n == 0:
            self.dif_loss = {}
            return

        # Set up for accumulation. Note that derivatives will be nonzero
        # only when the phi partials are nonzero.
        indices: typing.Set[int] = set().union(*[element.keys()
                                                 for element in self.dif_phi])
        self.dif_loss = {index: np.float64(0.) for index in indices}

        sum_phi = sum(self.phi)
        sum_t = sum(self.t)
        phi_phi = self.phi @ self.phi
        phi_t = self.phi @ self.t
        delta = n * phi_phi - sum_phi * sum_phi
        centered_t = self.t - sum_t / n
        n_var_t = centered_t @ centered_t

        # Compute some helpful vectors. We have, roughly,
        #   dif_beta = dif_beta_factor @ dif_phi
        # We say "roughly" here because dif_phi isn't stored in the
        # right format to do a dot product simply.
        dif_beta_0_factor = (2 * (sum_t - n * self.beta[0]) * self.phi
                             - sum_phi * self.t - phi_t) / delta
        dif_beta_1_factor = (n * (self.t - 2 * self.beta[1] * self.phi)
                             - sum_t) / delta

        # Accumulate
        twice_residual_sum = 2. * sum(self.residuals)
        twice_residual_phi = 2. * self.residuals @ self.phi
        for i, dif_phi_part in enumerate(self.dif_phi):
            twice_residual = 2. * self.residuals[i]
            for index, dif_phi in dif_phi_part.items():
                self.dif_loss[index] += (
                    twice_residual_sum * dif_beta_0_factor[i]
                    + twice_residual_phi * dif_beta_1_factor[i]
                    + twice_residual * self.beta[1]
                ) * dif_phi / n_var_t
