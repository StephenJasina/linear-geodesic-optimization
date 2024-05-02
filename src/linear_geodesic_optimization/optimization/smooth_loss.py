"""Module containing utilities to compute smoothness loss."""

import dcelmesh
import numpy as np
import numpy.typing as npt

from linear_geodesic_optimization.mesh.mesh import Mesh
from linear_geodesic_optimization.optimization.curvature \
    import Computer as Curvature
from linear_geodesic_optimization.optimization.laplacian \
    import Computer as Laplacian


class Computer:
    """Implementation of curvature loss."""

    def __init__(self, mesh: Mesh, laplacian: Laplacian, curvature: Curvature):
        """Initialize the computer."""
        self._mesh = mesh
        self._topology = mesh.get_topology()
        self._laplacian: Laplacian = laplacian
        self._curvature: Curvature = curvature

        # Forward variables
        self._forward_updates: int = mesh.get_updates() - 1
        self.loss: np.float64 = np.float64(0.)
        """
        The smoothness loss computed using the MVS-cross strategy.

        See
        http://graphics.berkeley.edu/papers/Joshi-EMC-2007-06/Joshi-EMC-2007-06.pdf
        for more information.
        """

        # Reverse variables
        self._reverse_updates: int = mesh.get_updates() - 1
        self._partials: npt.NDArray[np.float64] \
            = np.zeros((self._topology.n_vertices(), 3))
        self.dif_loss: npt.NDArray[np.float64] \
            = np.zeros(self._topology.n_vertices())
        """The partials of the smoothness loss, indexed by vertices."""

    def forward(self) -> None:
        """
        Compute the forward direction.

        The computed values will be stored in the variables:
        * `Computer.loss`
        """
        if self._forward_updates == self._mesh.get_updates():
            return
        self._laplacian.forward()
        self._curvature.forward()
        self._forward_updates = self._mesh.get_updates()

        self.loss = np.float64(0.)

        for edge, laplacian_element in zip(self._topology.edges(),
                                           self._laplacian.LC_dirichlet_edges):
            u, v = edge.vertices()
            self.loss -= 2 * laplacian_element \
                * self._curvature.kappa_1[u.index()] \
                * self._curvature.kappa_1[v.index()]
            self.loss -= 2 * laplacian_element \
                * self._curvature.kappa_2[u.index()] \
                * self._curvature.kappa_2[v.index()]

        for vertex, laplacian_element \
                in zip(self._topology.vertices(),
                       self._laplacian.LC_dirichlet_vertices):
            self.loss -= laplacian_element \
                * self._curvature.kappa_1[vertex.index()]**2
            self.loss -= laplacian_element \
                * self._curvature.kappa_2[vertex.index()]**2

        self.loss *= np.sum(self._laplacian.A)

    def reverse(self) -> None:
        """
        Compute the reverse direction (that is, partials).

        The computed values will be stored in the variables:
        * `Computer.dif_loss`
        """
        if self._reverse_updates == self._mesh.get_updates():
            return
        self.forward()
        self._laplacian.reverse()
        self._curvature.reverse()
        self._reverse_updates = self._mesh.get_updates()
        self._partials = self._mesh.get_partials()

        self.dif_loss = np.zeros(self._topology.n_vertices())
        for edge, laplacian_element, dif_laplacian_elements \
                in zip(self._topology.edges(),
                       self._laplacian.LC_dirichlet_edges,
                       self._laplacian.dif_LC_dirichlet_edges):
            u, v = edge.vertices()

            for index, dif_laplacian_element in dif_laplacian_elements.items():
                self.dif_loss[index] -= 2 * dif_laplacian_element \
                    * self._curvature.kappa_1[u.index()] \
                    * self._curvature.kappa_1[v.index()]
                self.dif_loss[index] -= 2 * dif_laplacian_element \
                    * self._curvature.kappa_2[u.index()] \
                    * self._curvature.kappa_2[v.index()]

            dif_kappa_1_elements_u = self._curvature.dif_kappa_1[u.index()]
            dif_kappa_2_elements_u = self._curvature.dif_kappa_2[u.index()]
            for index in dif_kappa_1_elements_u:
                dif_kappa_1_element_u = dif_kappa_1_elements_u[index]
                dif_kappa_2_element_u = dif_kappa_2_elements_u[index]
                self.dif_loss[index] -= 2 * laplacian_element \
                    * dif_kappa_1_element_u \
                    * self._curvature.kappa_1[v.index()]
                self.dif_loss[index] -= 2 * laplacian_element \
                    * dif_kappa_2_element_u \
                    * self._curvature.kappa_2[v.index()]

            dif_kappa_1_elements_v = self._curvature.dif_kappa_1[v.index()]
            dif_kappa_2_elements_v = self._curvature.dif_kappa_2[v.index()]
            for index in dif_kappa_1_elements_v:
                dif_kappa_1_element_v = dif_kappa_1_elements_v[index]
                dif_kappa_2_element_v = dif_kappa_2_elements_v[index]
                self.dif_loss[index] -= 2 * laplacian_element \
                    * self._curvature.kappa_1[u.index()] \
                    * dif_kappa_1_element_v
                self.dif_loss[index] -= 2 * laplacian_element \
                    * self._curvature.kappa_2[u.index()] \
                    * dif_kappa_2_element_v

        for vertex, laplacian_element, kappa_1, kappa_2, \
                dif_laplacian_elements, \
                dif_kappa_1_elements, dif_kappa_2_elements \
                in zip(self._topology.vertices(),
                       self._laplacian.LC_dirichlet_vertices,
                       self._curvature.kappa_1,
                       self._curvature.kappa_2,
                       self._laplacian.dif_LC_dirichlet_vertices,
                       self._curvature.dif_kappa_1,
                       self._curvature.dif_kappa_2):
            for index, dif_laplacian_element in dif_laplacian_elements.items():
                self.dif_loss[index] -= dif_laplacian_element \
                    * self._curvature.kappa_1[vertex.index()]**2
                self.dif_loss[index] -= dif_laplacian_element \
                    * self._curvature.kappa_2[vertex.index()]**2

            for index in dif_kappa_1_elements:
                dif_kappa_1_element = dif_kappa_1_elements[index]
                dif_kappa_2_element = dif_kappa_2_elements[index]
                self.dif_loss[index] -= 2. * laplacian_element \
                    * dif_kappa_1_element * kappa_1
                self.dif_loss[index] -= 2. * laplacian_element \
                    * dif_kappa_2_element * kappa_2

        total_area = np.sum(self._laplacian.A)

        self.dif_loss *= total_area

        for dif_A_part in self._laplacian.dif_A:
            for index, dif_A_element in dif_A_part.items():
                self.dif_loss[index] += self.loss * dif_A_element / total_area
