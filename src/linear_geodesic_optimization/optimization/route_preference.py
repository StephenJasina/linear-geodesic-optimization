"""Route preference loss for shortest paths on a height-field mesh.

The combinatorial paths are selected with Dijkstra on the current surface.
Their lengths are then recomputed with Torch, so a selected pair of paths
has a useful gradient with respect to mesh heights. Path changes make the
objective piecewise smooth.
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import torch


@dataclass(frozen=True)
class RouteConstraint:
    """A desired route through network vertex indices, with relative weight."""

    route: tuple[int, ...]
    weight: float = 1.0


class RoutePreference:
    """Prefer free shortest paths that stay near specified route polylines."""

    def __init__(self, mesh, network_vertices, constraints, corridor_width, margin):
        if not 0 < corridor_width < 1:
            raise ValueError('route_corridor_width must be between 0 and 1')
        if not np.isfinite(margin) or margin < 0:
            raise ValueError('route_margin must be finite and nonnegative')
        if not constraints:
            raise ValueError('route_constraints must contain at least one route')

        self._xy = np.asarray(mesh.get_coordinates()[:, :2], dtype=np.float64).copy()
        self._xy_torch = torch.as_tensor(self._xy)
        self._network_xy = np.asarray(network_vertices, dtype=np.float64)[:, :2]
        self._n_vertices = len(self._xy)
        self._edges = np.asarray([
            [vertex.index for vertex in edge.vertices()]
            for edge in mesh.get_topology().edges()
        ], dtype=np.int64)
        self._edge_xy_squared = np.sum(
            (self._xy[self._edges[:, 0]] - self._xy[self._edges[:, 1]]) ** 2,
            axis=1,
        )
        self._margin = float(margin)
        self._constraints = []
        for constraint in constraints:
            route = tuple(constraint.route)
            if len(route) < 2 or route[0] == route[-1]:
                raise ValueError('Each route must have distinct endpoints')
            if any(index < 0 or index >= len(self._network_xy) for index in route):
                raise ValueError('Route references an unknown network vertex')
            if not np.isfinite(constraint.weight) or constraint.weight <= 0:
                raise ValueError('Route weight must be finite and positive')
            waypoints = self._network_xy[list(route)]
            start = int(np.argmin(np.sum((self._xy - waypoints[0]) ** 2, axis=1)))
            end = int(np.argmin(np.sum((self._xy - waypoints[-1]) ** 2, axis=1)))
            if start == end:
                raise ValueError('Route endpoints map to the same mesh vertex')
            mask = self._corridor_mask(waypoints, corridor_width * mesh.get_scale())
            mask[start] = mask[end] = True
            allowed_edges = mask[self._edges[:, 0]] & mask[self._edges[:, 1]]
            if not np.any(allowed_edges):
                raise ValueError('Route corridor has no mesh edges; increase route_corridor_width')
            baseline = float(np.linalg.norm(self._xy[start] - self._xy[end]))
            self._constraints.append((route, float(constraint.weight), start, end,
                                      mask, allowed_edges, baseline))
        self._weight_sum = sum(item[1] for item in self._constraints)

    def _corridor_mask(self, waypoints, radius):
        distances_squared = np.full(self._n_vertices, np.inf)
        for a, b in zip(waypoints[:-1], waypoints[1:]):
            delta = b - a
            denominator = float(delta @ delta)
            if denominator == 0:
                continue
            fraction = np.clip((self._xy - a) @ delta / denominator, 0.0, 1.0)
            closest = a + fraction[:, None] * delta
            distances_squared = np.minimum(
                distances_squared, np.sum((self._xy - closest) ** 2, axis=1)
            )
        return distances_squared <= radius**2

    def _graph(self, edge_weights, allowed_edges=None):
        edges = self._edges if allowed_edges is None else self._edges[allowed_edges]
        weights = edge_weights if allowed_edges is None else edge_weights[allowed_edges]
        return csr_matrix((np.concatenate((weights, weights)),
                           (np.concatenate((edges[:, 0], edges[:, 1])),
                            np.concatenate((edges[:, 1], edges[:, 0])))),
                          shape=(self._n_vertices, self._n_vertices))

    @staticmethod
    def _path(predecessors, start, end):
        path = [end]
        while path[-1] != start:
            previous = int(predecessors[path[-1]])
            if previous < 0:
                return None
            path.append(previous)
        path.reverse()
        return np.asarray(path, dtype=np.int64)

    def _paths(self, heights):
        heights = np.asarray(heights, dtype=np.float64)
        edge_heights = heights[self._edges[:, 0]] - heights[self._edges[:, 1]]
        weights = np.sqrt(self._edge_xy_squared + edge_heights**2)
        full_graph = self._graph(weights)
        for route, weight, start, end, mask, allowed_edges, baseline in self._constraints:
            _, full_predecessors = dijkstra(
                full_graph, directed=False, indices=start, return_predecessors=True
            )
            free_path = self._path(full_predecessors, start, end)
            _, corridor_predecessors = dijkstra(
                self._graph(weights, allowed_edges), directed=False,
                indices=start, return_predecessors=True,
            )
            corridor_path = self._path(corridor_predecessors, start, end)
            if corridor_path is None:
                raise ValueError(
                    f'No mesh path through route {route}; increase route_corridor_width'
                )
            yield route, weight, free_path, corridor_path, mask, baseline

    def _path_length(self, path, heights):
        indices = torch.as_tensor(path, dtype=torch.long)
        u, v = indices[:-1], indices[1:]
        xy_difference = self._xy_torch[u] - self._xy_torch[v]
        height_difference = heights[u] - heights[v]
        return torch.sqrt(
            (xy_difference**2).sum(dim=1) + height_difference**2
        ).sum()

    def loss(self, heights):
        """Return a dimensionless weighted route-ranking loss."""
        total = heights.sum() * 0.0
        for _, weight, free_path, corridor_path, mask, baseline in self._paths(
            heights.detach().numpy()
        ):
            if free_path is None or np.all(mask[free_path]):
                continue
            free_length = self._path_length(free_path, heights)
            corridor_length = self._path_length(corridor_path, heights)
            violation = self._margin + (corridor_length - free_length) / baseline
            total = total + weight * torch.relu(violation) ** 2
        return total / self._weight_sum

    def metrics(self, heights):
        """Measure free-route fit without forcing intermediate waypoints."""
        height_tensor = torch.as_tensor(np.asarray(heights, dtype=np.float64))
        result = []
        for route, _, free_path, corridor_path, mask, baseline in self._paths(heights):
            free_length = float(self._path_length(free_path, height_tensor))
            corridor_length = float(self._path_length(corridor_path, height_tensor))
            result.append({
                'route': list(route),
                'free_path': free_path.tolist(),
                'fraction_free_path_in_corridor': float(np.mean(mask[free_path])),
                'free_length': free_length,
                'corridor_length': corridor_length,
                'relative_length_gap': (corridor_length - free_length) / baseline,
            })
        return result

    def guided_path(self, heights, constraint_index=0, precision=1e-3):
        """Find the weakest outside-corridor penalty yielding an inside path.

        This is a shortest path for weighted mesh edges, not an intrinsic
        geodesic of the unmodified height surface. Its edge cost is the
        surface length times (1 + strength) outside the route corridor.
        """
        if not 0 <= constraint_index < len(self._constraints):
            raise IndexError('Unknown route constraint')
        if precision <= 0:
            raise ValueError('precision must be positive')
        route, _, start, end, mask, allowed_edges, _ = self._constraints[constraint_index]
        heights = np.asarray(heights, dtype=np.float64)
        edge_height = heights[self._edges[:, 0]] - heights[self._edges[:, 1]]
        surface_lengths = np.sqrt(self._edge_xy_squared + edge_height**2)
        outside_edges = ~allowed_edges

        def shortest(strength):
            costs = surface_lengths * (1.0 + strength * outside_edges)
            distances, predecessors = dijkstra(
                self._graph(costs), directed=False, indices=start,
                return_predecessors=True,
            )
            path = self._path(predecessors, start, end)
            return path, float(distances[end])

        free_path, free_cost = shortest(0.0)
        if np.all(mask[free_path]):
            return {
                'route': list(route), 'strength': 0.0,
                'free_path': free_path.tolist(),
                'guided_path': free_path.tolist(),
                'free_corridor_fraction': 1.0,
                'guided_corridor_fraction': 1.0,
                'free_surface_length': free_cost,
                'guided_surface_length': free_cost,
                'guided_weighted_length': free_cost,
            }

        lower, upper = 0.0, 1.0
        guided_path, _ = shortest(upper)
        while not np.all(mask[guided_path]) and upper < 1e6:
            lower, upper = upper, 2.0 * upper
            guided_path, _ = shortest(upper)
        if not np.all(mask[guided_path]):
            raise ValueError(
                f'No mesh path through route {route}; increase route_corridor_width'
            )

        while upper - lower > precision:
            midpoint = (lower + upper) / 2.0
            candidate, _ = shortest(midpoint)
            if np.all(mask[candidate]):
                upper, guided_path = midpoint, candidate
            else:
                lower = midpoint
        # Add a small buffer to avoid numerical ties changing the path.
        strength = upper + precision
        guided_path, weighted_cost = shortest(strength)
        guided_surface_length = float(
            self._path_length(guided_path, torch.as_tensor(heights))
        )
        return {
            'route': list(route),
            'strength': float(strength),
            'free_path': free_path.tolist(),
            'guided_path': guided_path.tolist(),
            'free_corridor_fraction': float(np.mean(mask[free_path])),
            'guided_corridor_fraction': float(np.mean(mask[guided_path])),
            'free_surface_length': float(free_cost),
            'guided_surface_length': float(guided_surface_length),
            'guided_weighted_length': weighted_cost,
        }
