"""Module containing utilities to compute curvature loss."""

import typing

import numpy as np
import numpy.typing as npt

from linear_geodesic_optimization.mesh.mesh import Mesh
from linear_geodesic_optimization.optimization.curvature \
    import Computer as Curvature


def tapered_tube(
    points: npt.NDArray[np.float64],
    network_vertices: npt.NDArray[np.float64],
    network_edges: typing.List[typing.Tuple[int, int]],
    network_curvatures: typing.List[typing.Optional[float]],
    epsilon: float,
    falloff: float,
    falloff_weight: bool = False,
    falloff_ends: bool = True,
) -> typing.List[typing.Tuple[npt.NDArray[np.int64], npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
    """
    Return, per network edge, `(indices, targets, weights)` for a tube
    with a soft edge.

    Vertices within `epsilon` of the edge (as a segment) have the edge's
    curvature as their target. In the ring out to `epsilon + falloff`,
    the target fades to 0 as `curvature * b(d)` with the cosine taper
    `b(d) = (1 + cos(pi (d - epsilon) / falloff)) / 2`, so the loss
    asks the curvature to fade out instead of leaving the ring free
    (which otherwise collects the compensating curvature in a rim just
    outside the tube). Each vertex weighs 1, or `b(d)` when
    `falloff_weight` is true, and the weights are normalized to sum to
    1 per edge. Edges with no target or no vertices get empty arrays.

    With `falloff_ends` false, the ring only runs along the sides of the
    edge (vertices projecting strictly inside the segment), not around
    its ends, so it doesn't reach over the nodes, where edges with other
    targets meet.
    """
    points = np.asarray(points, dtype=np.float64)[:, :2]
    network_vertices = np.asarray(network_vertices, dtype=np.float64)
    result = []
    for (i, j), curvature in zip(network_edges, network_curvatures):
        a = network_vertices[i, :2]
        direction = network_vertices[j, :2] - a
        length_squared = direction @ direction
        if length_squared:
            fraction = np.clip(((points - a) @ direction) / length_squared, 0., 1.)
        else:
            fraction = np.zeros(len(points))
        distance = np.linalg.norm(points - (a + fraction[:, None] * direction), axis=1)
        within = distance < epsilon + falloff
        if not falloff_ends:
            within &= (distance < epsilon) | ((fraction > 0.) & (fraction < 1.))
        members = np.flatnonzero(within)
        if curvature is None or len(members) == 0:
            result.append((np.zeros(0, dtype=np.int64), np.zeros(0), np.zeros(0)))
            continue
        ring = np.clip((distance[members] - epsilon) / falloff, 0., 1.) if falloff > 0 \
            else np.zeros(len(members))
        taper = 0.5 * (1. + np.cos(np.pi * ring))
        weights = taper if falloff_weight else np.ones(len(members))
        result.append((members, curvature * taper, weights / weights.sum()))
    return result


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
        edge_weights = None,
        falloff: float = 0.,
        falloff_weight: bool = False,
        falloff_ends: bool = True,
    ):
        """
        Initialize the computer. With a positive `falloff`, the tube
        has a soft edge (see `tapered_tube`).
        """
        self._mesh = mesh
        self._topology = mesh.get_topology()
        self._curvature: Curvature = curvature
        self._tapered = None
        if falloff > 0:
            self._tapered = tapered_tube(
                mesh.get_coordinates()[:, :2], network_vertices,
                network_edges, network_curvatures, epsilon, falloff,
                falloff_weight, falloff_ends,
            )

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
            edge_weights = np.array([1.] * len(network_edges))
        edge_weights = edge_weights * edge_lengths
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
            = np.zeros((self._topology.n_vertices, 3))
        self.dif_loss: npt.NDArray[np.float64] \
            = np.zeros(self._topology.n_vertices)
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
        if self._tapered is not None:
            for (indices, targets, weights), edge_weight in zip(self._tapered, self._edge_weights):
                if len(indices):
                    residuals = np.asarray(self._curvature.kappa_G)[indices] - targets
                    self.loss += edge_weight * np.sum(weights * residuals**2)
            return
        for fat_edge, network_curvature, edge_weight in zip(
            self._fat_edges,
            self._network_curvatures,
            self._edge_weights
        ):
            if network_curvature is None or len(fat_edge) == 0:
                continue

            loss_part = np.float64(0.)
            for vertex in fat_edge:
                loss_part += (self._curvature.kappa_G[vertex.index]
                              - network_curvature)**2

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

        self.dif_loss = np.zeros(self._topology.n_vertices)
        if self._tapered is not None:
            for (indices, targets, weights), edge_weight in zip(self._tapered, self._edge_weights):
                for vertex, target, weight in zip(indices, targets, weights):
                    coefficient = 2 * edge_weight * weight * (self._curvature.kappa_G[vertex] - target)
                    for index, partial in self._curvature.dif_kappa_G[vertex].items():
                        self.dif_loss[index] += coefficient * partial
            return
        for fat_edge, network_curvature, edge_weight \
                in zip(self._fat_edges,
                       self._network_curvatures,
                       self._edge_weights):
            if network_curvature is None or len(fat_edge) == 0:
                continue

            for vertex in fat_edge:
                curvature = self._curvature.kappa_G[vertex.index]
                for index, partial \
                        in self._curvature.dif_kappa_G[vertex.index].items():
                    self.dif_loss[index] \
                        += 2 * edge_weight * (curvature - network_curvature) \
                        * partial / len(fat_edge)
