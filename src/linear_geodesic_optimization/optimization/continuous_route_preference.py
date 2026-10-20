"""Route loss using exact geodesics on the triangulated height surface."""

import numpy as np
from pygeodesic import geodesic
from scipy.spatial import cKDTree
import torch


class ContinuousRoutePreference:
    """Compare unrestricted and route-corridor continuous surface geodesics."""

    def __init__(self, mesh, network_vertices, constraints, corridor_width, margin):
        if not 0 < corridor_width < 1:
            raise ValueError('route_corridor_width must be between 0 and 1')
        if not np.isfinite(margin) or margin < 0:
            raise ValueError('route_margin must be finite and nonnegative')
        if not constraints:
            raise ValueError('route_constraints must contain at least one route')
        self._xy = np.asarray(mesh.get_coordinates()[:, :2], dtype=np.float64).copy()
        self._faces = np.asarray([
            [vertex.index for vertex in face.vertices()]
            for face in mesh.get_topology().faces()
        ], dtype=np.int32)
        self._network_xy = np.asarray(network_vertices, dtype=np.float64)[:, :2]
        self._scale = mesh.get_scale()
        triangle = self._xy[self._faces]
        self._triangle_origins = triangle[:, 0]
        matrix = np.stack((triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]), axis=2)
        self._inverse_triangles = np.linalg.inv(matrix)
        self._face_tree = cKDTree(triangle.mean(axis=1))
        self._constraints = []
        for constraint in constraints:
            route = tuple(constraint.route)
            if len(route) < 2 or route[0] == route[-1]:
                raise ValueError('Each route needs distinct endpoints')
            if any(index < 0 or index >= len(self._network_xy) for index in route):
                raise ValueError('Route references an unknown network vertex')
            if not np.isfinite(constraint.weight) or constraint.weight <= 0:
                raise ValueError('Route weight must be finite and positive')
            waypoints = self._network_xy[list(route)]
            start, end = [
                int(np.argmin(np.sum((self._xy - waypoint) ** 2, axis=1)))
                for waypoint in (waypoints[0], waypoints[-1])
            ]
            if start == end:
                raise ValueError('Route endpoints map to the same mesh vertex')
            distances = self._distances_to_route(self._xy, waypoints)
            mask = distances <= corridor_width * self._scale
            mask[start] = mask[end] = True
            subfaces = self._faces[np.all(mask[self._faces], axis=1)]
            used = np.unique(subfaces)
            remap = np.full(len(self._xy), -1, dtype=np.int32)
            remap[used] = np.arange(len(used), dtype=np.int32)
            if remap[start] < 0 or remap[end] < 0:
                raise ValueError('Route corridor does not connect the endpoints')
            self._constraints.append({
                'route': route,
                'weight': float(constraint.weight),
                'waypoints': waypoints,
                'start': start,
                'end': end,
                'used': used,
                'faces': remap[subfaces],
                'sub_start': int(remap[start]),
                'sub_end': int(remap[end]),
                'baseline': float(np.linalg.norm(self._xy[start] - self._xy[end])),
                'width': corridor_width * self._scale,
            })
        self._weight_sum = sum(item['weight'] for item in self._constraints)
        self._margin = float(margin)

    @staticmethod
    def _distances_to_route(points, waypoints):
        distances = np.full(len(points), np.inf)
        for start, end in zip(waypoints[:-1], waypoints[1:]):
            direction = end - start
            denominator = float(direction @ direction)
            if denominator == 0:
                continue
            fraction = np.clip(((points - start) @ direction) / denominator, 0, 1)
            distances = np.minimum(
                distances,
                np.linalg.norm(points - (start + fraction[:, None] * direction), axis=1),
            )
        return distances

    def _paths(self, heights):
        points = np.column_stack((self._xy, np.asarray(heights, dtype=np.float64)))
        free_solver = geodesic.PyGeodesicAlgorithmExact(points, self._faces)
        for constraint in self._constraints:
            free_distance, free_path = free_solver.geodesicDistance(
                constraint['start'], constraint['end']
            )
            restricted = geodesic.PyGeodesicAlgorithmExact(
                points[constraint['used']], constraint['faces']
            )
            corridor_distance, corridor_path = restricted.geodesicDistance(
                constraint['sub_start'], constraint['sub_end']
            )
            if (not np.isfinite(free_distance) or not np.isfinite(corridor_distance)
                    or len(free_path) < 2 or len(corridor_path) < 2):
                raise ValueError('Route corridor is disconnected; increase route_corridor_width')
            yield constraint, free_path, corridor_path, free_distance, corridor_distance

    def _length(self, path, heights):
        xy = np.asarray(path[:, :2], dtype=np.float64)
        candidates = np.atleast_2d(self._face_tree.query(xy, k=min(12, len(self._faces)))[1])
        origins = self._triangle_origins[candidates]
        inverse = self._inverse_triangles[candidates]
        coordinates = np.einsum('nkij,nkj->nki', inverse, xy[:, None, :] - origins)
        weights = np.concatenate((1 - coordinates.sum(axis=2, keepdims=True), coordinates), axis=2)
        choice = np.argmax(weights.min(axis=2), axis=1)
        chosen_weights = weights[np.arange(len(xy)), choice]
        if np.min(chosen_weights) < -1e-5:
            raise ValueError('Cannot locate exact geodesic on mesh faces')
        face_indices = torch.as_tensor(self._faces[candidates[np.arange(len(xy)), choice]], dtype=torch.long)
        barycentric = torch.as_tensor(chosen_weights, dtype=heights.dtype)
        path_heights = (heights[face_indices] * barycentric).sum(dim=1)
        horizontal = torch.as_tensor(np.diff(xy, axis=0), dtype=heights.dtype)
        vertical = path_heights[1:] - path_heights[:-1]
        return torch.sqrt((horizontal**2).sum(dim=1) + vertical**2).sum()

    def loss(self, heights):
        total = heights.sum() * 0.0
        for constraint, free_path, corridor_path, _, _ in self._paths(heights.detach().numpy()):
            free_length = self._length(free_path, heights)
            corridor_length = self._length(corridor_path, heights)
            violation = self._margin + (corridor_length - free_length) / constraint['baseline']
            total = total + constraint['weight'] * torch.relu(violation) ** 2
        return total / self._weight_sum

    def metrics(self, heights):
        result = []
        for constraint, free_path, corridor_path, free_distance, corridor_distance in self._paths(heights):
            segments = free_path[1:] - free_path[:-1]
            lengths = np.linalg.norm(segments, axis=1)
            sample_t = (np.arange(16) + .5) / 16
            samples = (free_path[:-1, None, :] + sample_t[None, :, None] * segments[:, None, :]).reshape(-1, 3)
            sample_lengths = np.repeat(lengths / 16, 16)
            inside = self._distances_to_route(samples[:, :2], constraint['waypoints']) <= constraint['width']
            result.append({
                'route': list(constraint['route']),
                'method': 'continuous',
                'free_path_xy': (free_path[:, :2] / self._scale).tolist(),
                'fraction_free_path_in_corridor': float(sample_lengths[inside].sum() / sample_lengths.sum()),
                'free_length': float(free_distance),
                'corridor_length': float(corridor_distance),
                'relative_length_gap': float((corridor_distance - free_distance) / constraint['baseline']),
            })
        return result
