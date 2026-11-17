"""
A loss that makes network routes the shortest paths on the surface.

For each route, compare the surface length of its lift (the polyline
through its waypoints) with the length of the shortest path between its
endpoints that collation.py draws: a Dijkstra trace through the route's
waypoints on the mesh graph, tightened into a geodesic by
`potpourri3d`'s edge-flip solver. The geodesic is free to leave the
waypoints, so it is shorter than the route whenever the surface offers a
shortcut. With route length L and geodesic length G, the loss is

    sum_r w_r max(L_r / G_r - 1, 0)^2,

with the route weights w_r (traffic) normalized to sum to 1. It is 0
exactly when no route is longer than its geodesic.

The geodesics ("competitors") are held fixed between calls to
`refresh`. Because a geodesic is a critical point of length, the
gradient of its fixed-path length is the gradient of the geodesic
distance, so the loss gradient is right at the heights where the
competitors were computed. Optimization therefore alternates between
refreshing the competitors and taking L-BFGS steps.

Lengths are integrals of sqrt(1 + h_t^2) over the planar curves, with
`h_t` from `geodesic_curvature.directional_operators`. They are measured
on the heights blurred by a Gaussian of width `smoothing`. Without the
blur, the gradient is concentrated on the (fixed) competitor curves, and
the optimizer lengthens them by etching grid-scale grooves under them
rather than by shaping broad valleys.
"""

import types
import typing

import numpy as np
import numpy.typing as npt
import potpourri3d as pp3d
import scipy.sparse
import scipy.sparse.csgraph

from linear_geodesic_optimization import geodesic_curvature as gc
from linear_geodesic_optimization.data import utility
from linear_geodesic_optimization.mesh.rectangle import Mesh


class _SampledCurves(typing.NamedTuple):
    operator: scipy.sparse.csr_array
    """Maps heights to h_t at the samples."""
    planar_lengths: npt.NDArray[np.float64]
    curve: npt.NDArray[np.int64]
    """Index of the curve each sample belongs to."""


def _sample_curves(mesh, curves, spacing) -> _SampledCurves:
    positions, tangents, lengths, indices = [], [], [], []
    for index, points in enumerate(curves):
        points = np.asarray(points, dtype=np.float64)[:, :2]
        points = points[np.concatenate((
            [True], np.linalg.norm(np.diff(points, axis=0), axis=1) > 0
        ))]
        samples = gc.polyline_samples(points, spacing)
        positions.append(samples.positions)
        tangents.append(samples.tangents)
        lengths.append(samples.planar_lengths)
        indices.append(np.full(len(samples.positions), index))
    keep, (a_t, _, _) = gc.directional_operators(
        mesh, np.concatenate(positions), np.concatenate(tangents)
    )
    return _SampledCurves(
        a_t, np.concatenate(lengths)[keep], np.concatenate(indices)[keep]
    )


def _lengths(sampled: _SampledCurves, z, n_curves):
    """Return the surface lengths and the per-sample d(length)/d(h_t)."""
    h_t = sampled.operator @ z
    speed = np.sqrt(1 + h_t**2)
    lengths = np.bincount(
        sampled.curve, sampled.planar_lengths * speed, minlength=n_curves
    )
    return lengths, sampled.planar_lengths * h_t / speed


def gaussian_blur_matrix(sides: int, width: float) -> npt.NDArray[np.float64]:
    """
    Return the `sides` by `sides` matrix of a 1D Gaussian blur of
    standard deviation `width` (in grid spacings), with rows normalized
    to sum to 1. A width of 0 gives the identity.
    """
    if width <= 0:
        return np.eye(sides)
    offsets = np.arange(sides)
    blur = np.exp(-0.5 * ((offsets[:, None] - offsets[None, :]) / width)**2)
    blur[blur < 1e-8] = 0.
    return blur / blur.sum(axis=1, keepdims=True)


class RouteStretch:
    """
    The route stretch loss for `routes`, each an array of planar
    waypoint coordinates, weighted by `weights`. `spacing` is the sample
    spacing used to integrate lengths, and `smoothing` the standard
    deviation of the Gaussian blur applied to the heights first (both in
    the mesh's units).
    """

    def __init__(
        self,
        mesh: Mesh,
        routes: typing.Sequence[npt.NDArray[np.float64]],
        weights: npt.NDArray[np.float64],
        spacing: float,
        smoothing: float = 0.,
    ):
        self.mesh = mesh
        self.routes = [
            np.asarray(route, dtype=np.float64)[:, :2] for route in routes
        ]
        weights = np.asarray(weights, dtype=np.float64)
        self.weights = weights / weights.sum()
        self.spacing = spacing
        self._routes_sampled = _sample_curves(mesh, self.routes, spacing)
        self.competitors = None
        self._competitors_sampled = None

        sides = mesh.get_width()
        self._sides = sides
        self._grid_spacing = mesh.get_scale() / (sides - 1)
        self._blur = gaussian_blur_matrix(sides, smoothing / self._grid_spacing)
        topology = mesh.get_topology()
        self._faces = np.array([
            [vertex.index for vertex in face.vertices()]
            for face in topology.faces()
        ])
        self._edges = np.array([
            [vertex.index for vertex in edge.vertices()]
            for edge in topology.edges()
        ])
        xy = mesh.get_coordinates()[:, :2]
        # Snap each waypoint to its nearest grid vertex
        self._waypoint_vertices = []
        for route in self.routes:
            ij = np.clip(np.rint(route / self._grid_spacing).astype(int),
                         0, sides - 1)
            vertices = ij[:, 0] * sides + ij[:, 1]
            if not np.allclose(xy[vertices], ij * self._grid_spacing):
                raise ValueError('RouteStretch needs an untrimmed rectangle mesh')
            self._waypoint_vertices.append(vertices)

    def refresh(self, z: npt.NDArray[np.float64]) -> None:
        """Recompute the competing geodesics on the surface with heights `z`."""
        xy = self.mesh.get_coordinates()[:, :2]
        points = np.column_stack((xy, z))
        u, v = self._edges.T
        weights = np.linalg.norm(points[u] - points[v], axis=1)
        n = len(points)
        graph = scipy.sparse.csr_array(
            (weights, (u, v)), shape=(n, n)
        )
        sources = np.unique(np.concatenate([
            vertices[:-1] for vertices in self._waypoint_vertices
        ]))
        _, predecessors = scipy.sparse.csgraph.dijkstra(
            graph, directed=False, indices=sources, return_predecessors=True
        )
        row_of = {source: row for row, source in enumerate(sources)}
        solver = pp3d.EdgeFlipGeodesicSolver(points, self._faces)

        competitors = []
        for route, vertices in zip(self.routes, self._waypoint_vertices):
            trace = []
            for source, target in zip(vertices[:-1], vertices[1:]):
                path = [target]
                while path[-1] != source:
                    path.append(predecessors[row_of[source], path[-1]])
                path = path[::-1]
                trace.extend(path if not trace else path[1:])
            trace = utility.remove_loops([int(vertex) for vertex in trace])
            if len(trace) < 2:
                competitor = route[[0, -1]]
            else:
                competitor = solver.find_geodesic_path_poly(trace)[:, :2]
                # Start and end exactly at the route's endpoints rather
                # than at the snapped grid vertices
                competitor = np.concatenate(
                    (route[:1], competitor[1:-1], route[-1:])
                )
            competitors.append(competitor)
        self.competitors = competitors
        self._competitors_sampled = _sample_curves(
            self.mesh, competitors, self.spacing
        )

    def _smooth(self, z):
        z = np.asarray(z, dtype=np.float64).reshape(self._sides, self._sides)
        return (self._blur @ z @ self._blur.T).reshape(-1)

    def _smooth_adjoint(self, gradient):
        gradient = gradient.reshape(self._sides, self._sides)
        return (self._blur.T @ gradient @ self._blur).reshape(-1)

    def stretches(self, z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return L_r / G_r for each route, against the current competitors."""
        z = self._smooth(z)
        n = len(self.routes)
        route_lengths, _ = _lengths(self._routes_sampled, z, n)
        competitor_lengths, _ = _lengths(self._competitors_sampled, z, n)
        return route_lengths / competitor_lengths

    def evaluate(
        self, z: npt.NDArray[np.float64]
    ) -> typing.Tuple[float, npt.NDArray[np.float64]]:
        """Return the loss and its gradient with respect to `z`."""
        if self._competitors_sampled is None:
            raise RuntimeError('Call refresh before evaluate')
        z = self._smooth(z)
        n = len(self.routes)
        route_lengths, route_slopes = _lengths(self._routes_sampled, z, n)
        competitor_lengths, competitor_slopes = _lengths(
            self._competitors_sampled, z, n
        )
        excess = np.maximum(route_lengths / competitor_lengths - 1, 0.)
        loss = float(self.weights @ excess**2)
        coefficient = 2 * self.weights * excess
        d_route = coefficient / competitor_lengths
        d_competitor = -coefficient * route_lengths / competitor_lengths**2
        gradient = self._routes_sampled.operator.T @ (
            d_route[self._routes_sampled.curve] * route_slopes
        ) + self._competitors_sampled.operator.T @ (
            d_competitor[self._competitors_sampled.curve] * competitor_slopes
        )
        return loss, self._smooth_adjoint(gradient)


class WithRouteStretch:
    """
    A loss computer (either backend) plus `lambda_stretch` times a
    `RouteStretch` loss, with the same interface.
    """

    def __init__(self, computer, stretch: RouteStretch, lambda_stretch: float):
        self.computer = computer
        self.stretch = stretch
        self.lambda_stretch = lambda_stretch
        self.stretch_loss = types.SimpleNamespace(loss=0.)

    def __getattr__(self, name):
        return getattr(self.computer, name)

    def _stretch(self, z):
        z = self.computer.mesh.get_parameters() if z is None else np.asarray(z)
        loss, gradient = self.stretch.evaluate(z)
        self.stretch_loss.loss = loss
        return loss, gradient

    def forward(self, z=None):
        loss = self.computer.forward(z)
        return loss + self.lambda_stretch * self._stretch(z)[0]

    def reverse(self, z=None):
        gradient = self.computer.reverse(z)
        return gradient + self.lambda_stretch * self._stretch(z)[1]

    def diagnostics(self, z=None, f=None, context=None):
        loss, _ = self._stretch(z)
        print(f'\tL_stretch: {loss:.6f}')
        return self.computer.diagnostics(z, f, context)

