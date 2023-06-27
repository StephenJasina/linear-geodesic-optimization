"""Functions for computing quantities related to the convex hull."""

import typing

import numpy as np
import numpy.typing as npt


def get_connected_components(
    n: int,
    edges: typing.List[typing.Tuple[int, int]]
) -> typing.List[typing.List[int]]:
    """
    Compute the connected components of a graph.

    As input, take the number of vertices and a list of tuples
    representing edges. Return a list of connected components, where
    each component is represented as a list of indices.
    """
    adjacency_list: typing.List[typing.List[int]] = [[] for _ in range(n)]
    for (i, j) in edges:
        adjacency_list[i].append(j)
        adjacency_list[j].append(i)
    visited = [False for _ in range(n)]

    components: typing.List[typing.List[int]] = []
    for i in range(n):
        if visited[i]:
            continue

        component: typing.List[int] = []
        stack = [i]
        while stack:
            j = stack[-1]
            del stack[-1]

            if visited[j]:
                continue

            component.append(j)
            stack.extend(adjacency_list[j])
            visited[j] = True
        components.append(component)

    return components


def compute_convex_hull(
    points: npt.NDArray[np.float64],
    indices: typing.Optional[typing.List[int]] = None
) -> typing.List[int]:
    """
    Run the Graham scan algorithm.

    As input, take a list of coordinate pairs and an optional list of
    indices for which to subset the coordinates (useful for looking only
    at a connected component, for example).

    Return a list of the indices of the vertices on the convex hull,
    oriented counter-clockwise.
    """
    if indices is None:
        indices = list(range(len(points)))

    pivot_point_index = 0
    pivot_point = points[pivot_point_index]
    for index in indices:
        point = points[index]
        if point[1] < pivot_point[1] or (point[1] == pivot_point[1]
                                         and point[0] < pivot_point[0]):
            pivot_point_index = index
            pivot_point = point

    points = points - pivot_point

    sorted_indices = [index for _, index in sorted(
        (
            (
                -point[0] / np.linalg.norm(point),
                np.linalg.norm(point)
            ),
            index
        )
        for index in indices
        for point in (points[index],)  # This is essentially a let
        if point @ point > 0
    )]

    convex_hull = [pivot_point_index]
    for index in sorted_indices:
        point_index = points[index]
        while True:
            if len(convex_hull) <= 1:
                break

            point_top = points[convex_hull[-1]]
            point_second_to_top = points[convex_hull[-2]]

            if np.cross(point_top - point_index,
                        point_second_to_top - point_index) < 0:
                break

            del convex_hull[-1]

        convex_hull.append(index)

    return convex_hull


def compute_connected_convex_hulls(
    points: npt.NDArray[np.float64],
    edges: typing.List[typing.Tuple[int, int]]
) -> typing.List[typing.List[int]]:
    """
    Compute the convex hull of each connected component of a graph.

    This function returns a list of the convex hulls, where each convex
    hull is represented as a list of indices of boundary vertices
    oriented counterclockwise.
    """
    n = len(points)
    return [
        compute_convex_hull(points, component)
        for component in get_connected_components(n, edges)
    ]


def is_in_convex_hull(
    point: npt.NDArray[np.float64],
    points: npt.NDArray[np.float64],
    convex_hull: typing.List[int]
) -> bool:
    """
    Return whether a point lies in a convex hull.

    The convex hull should be the output as computed by
    `compute_convex_hull`.
    """
    if len(convex_hull) < 2:
        return False

    # The signs of these cross products determines whether the point is
    # on the interior or exterior of the convex hull
    crosses = np.array([
        np.cross(point - points[left_index],
                 points[right_index] - points[left_index])
        for left_index, right_index in zip(convex_hull,
                                           [*convex_hull[1:], convex_hull[0]])
    ])

    return np.all(crosses >= 0) or np.all(crosses <= 0)


def project_to_line_segment(
    point: npt.NDArray[np.float64],
    left: npt.NDArray[np.float64],
    right: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Project a point onto the line passing through left and right."""
    direction = right - left
    line_projection = left \
        + (point - left) @ direction / (direction @ direction) * direction
    if (line_projection - left) @ (right - left) >= 0 \
            and (line_projection - right) @ (left - right) >= 0:
        return line_projection

    if np.linalg.norm(point - left) <= np.linalg.norm(point - right):
        return np.array(left)
    else:
        return np.array(right)


def distance_to_convex_hulls(
    point: npt.NDArray[np.float64],
    points: npt.NDArray[np.float64],
    convex_hulls: typing.List[typing.List[int]]
) -> np.float64:
    """
    Return the minimal distance to a set of convex hulls.

    The convex hulls should be the outputs as computed by
    `compute_convex_hull`.
    """
    distance = np.float64(np.inf)
    for convex_hull in convex_hulls:
        if is_in_convex_hull(point, points, convex_hull):
            return np.float64(0.)

        if len(convex_hull) == 1:
            projections = [points[convex_hull[0]]]
        else:
            projections = [
                project_to_line_segment(point,
                                        points[left_index],
                                        points[right_index])
                for left_index, right_index
                in zip(convex_hull, [*convex_hull[1:], convex_hull[0]])
            ]
        distance = min(distance,
                       min([np.linalg.norm(point - projection)
                            for projection in projections]))
    return distance
