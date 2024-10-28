"""Module containing utilities to compute curvature loss."""

import typing

import numpy as np
import numpy.typing as npt

from linear_geodesic_optimization.mesh.mesh import Mesh
from linear_geodesic_optimization.optimization.curvature \
    import Computer as Curvature


class Computer:
    """Implementation of curvature loss."""

    def __init__(
        self,
        mesh: Mesh,
        network_vertices: npt.NDArray[np.float64],
        network_edges: typing.List[typing.Tuple[int, int]],
        network_curvatures: typing.List[np.float64],
        epsilon: np.float64,
        curvature: Curvature,
        edge_weights = None
    ):
        """Initialize the computer."""
        self._mesh = mesh
        self._topology = mesh.get_topology()
        self._curvature: Curvature = curvature

        self._fat_edges = mesh.get_fat_edges(
            network_vertices,
            network_edges,
            epsilon
        )

        edge_lengths = np.array([
            np.linalg.norm(network_vertices[i] - network_vertices[j])
            for i, j in network_edges
        ])
        if edge_weights is None:
            edge_weights = [1.] * len(network_edges)
        edge_weights = np.array(edge_weights) * edge_lengths
        self._edge_weights = edge_weights / sum(edge_weights)

        self._network_curvatures = network_curvatures

        # Forward variables
        self._forward_updates: int = mesh.get_updates() - 1
        self.loss: np.float64 = np.float64(0.)
        """
        The curvature loss.

        For each network edge passed in, estimate the integral of
        (kappa_G - desired_curvature)^2. Then, sum the integral
        approximations and divide by the total length.

        TODO: Improve this estimation by doing an actual integral.
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
        self._curvature.forward()
        self._forward_updates = self._mesh.get_updates()

        self.loss = np.float64(0.)
        for fat_edge, network_curvature, edge_weight in zip(
            self._fat_edges,
            self._network_curvatures,
            self._edge_weights
        ):
            if len(fat_edge) == 0:
                continue
            loss_part = np.float64(0.)
            for vertex in fat_edge:
                loss_part += (self._curvature.kappa_G[vertex.index]
                              - network_curvature)**2
            # print(network_curvature, edge_weight, loss_part * edge_weight / len(fat_edge))
            self.loss += loss_part * edge_weight / len(fat_edge)

    def reverse(self) -> None:
        """
        Compute the reverse direction (that is, partials).

        The computed values will be stored in the variables:
        * `Computer.dif_loss`
        """
        if self._reverse_updates == self._mesh.get_updates():
            return
        self.forward()
        self._curvature.reverse()
        self._reverse_updates = self._mesh.get_updates()
        self._partials = self._mesh.get_partials()

        self.dif_loss = np.zeros(self._topology.n_vertices())
        for fat_edge, network_curvature, edge_weight \
                in zip(self._fat_edges,
                       self._network_curvatures,
                       self._edge_weights):
            for vertex in fat_edge:
                curvature = self._curvature.kappa_G[vertex.index]
                for index, partial \
                        in self._curvature.dif_kappa_G[vertex.index].items():
                    self.dif_loss[index] \
                        += 2 * edge_weight * (curvature - network_curvature) \
                        * partial / len(fat_edge)
