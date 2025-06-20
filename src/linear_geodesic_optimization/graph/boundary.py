"""
Functions for computing quantities related to the graph boundary.

This includes connected components of, convex hulls of, and borders
around networks.
"""

import collections
import itertools
import math
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

    components = [
        component
        for component in components
        if len(component) > 1
    ]

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
    oriented counterclockwise.
    """
    if indices is None:
        indices = list(range(len(points)))

    pivot_point_index = indices[0]
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

    For this problem, we care only about the connected components that
    contain an edge, so each component will have at least two vertices.
    """
    n = len(points)
    return [
        convex_hull
        for component in get_connected_components(n, edges)
        for convex_hull in (compute_convex_hull(points, component),)
        if len(convex_hull) > 1
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
    if len(convex_hull) == 0:
        return False

    if len(convex_hull) == 1:
        return np.all(np.equal(point, points[convex_hull[0]]))

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
    """Project `point` onto the segment connecting `left` and `right`."""
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

def is_in_interior_of_polygons(
    point: npt.NDArray[np.float64],
    polygons: typing.Iterable[typing.List[npt.NDArray[np.float64]]]
):
    """
    Determine whether a point is in any of a collection of polygons.

    Polygons are stored as a list of their boundary points, oriented
    counterclockwise (though, this function should still work for those
    oriented in the other direction).
    """
    for polygon in polygons:
        # Check how many times the ray from point extending to the right
        # crosses the polygon. An even number of times means the point
        # is on the exterior; an odd number means interior
        is_in_interior = False

        # First, count the vertex crossings
        for vertex in polygon:
            if vertex[1] == point[1]:
                if vertex[0] > point[0]:
                    is_in_interior = not is_in_interior
                elif vertex[0] == point[0]:
                    # If the point is on the boundary, then it is in the
                    # interior
                    return True


        # Now count the edge crossings
        for u, v in itertools.islice(itertools.pairwise(itertools.cycle(polygon)), len(polygon)):
            # Skip the cases covered by the vertex crossings
            if u[1] == point[1] or v[1] == point[1]:
                continue

            if (u[1] > point[1]) != (v[1] > point[1]):
                # In this case, one endpoint lies above the ray, and the
                # other endpoint lies below. We need to figure out where
                # it crosses the horizontal line crossing through the
                # point
                alpha = (point[1] - u[1]) / (v[1] - u[1])
                intersection_x = u[0] + alpha * (v[0] - u[0])
                if intersection_x > point[0]:
                    is_in_interior = not is_in_interior
                elif intersection_x == point[0]:
                    # The point is on the boundary
                    return True

        if is_in_interior:
            return True

    return False

def compute_segment_segment_intersection(
    left1, right1,
    left2, right2
):
    """
    Compute the intersection between two line segments.

    The segments are between `left1` and `right1` and between `left2`
    and `right2`, ignoring their boundaries.

    If no intersection point exists, or if the line segments are
    parallel, return `None`.

    Otherwise, return the value alpha such that the intersection point
    is at left1 + alpha * (right1 - left1)
    """
    # Special cases for if endpoints are shared (to avoid floating
    # point errors)
    if np.all(left1 == left2) or np.all(left1 == right2):
        # return 0.
        return None
    if np.all(right1 == left2) or np.all(right1 == right2):
        # return 1.
        return None

    b1 = left1
    m1 = right1 - left1

    b2 = left2
    m2 = right2 - left2

    # Need to solve the equation b1 + alpha1 * m1 = b2 + alpha2 * m2
    # for numbers alpha1 and alpha2 both between 0 and 1. This is
    # equivalent to alpha1 * m1 - alpha2 * m2 = b2 - b1
    determinant = -m1[0] * m2[1] + m1[1] * m2[0]
    if determinant == 0:
        return None
    b = b2 - b1
    alpha1 = (-b[0] * m2[1] + b[1] * m2[0]) / determinant
    alpha2 = (-b[0] * m1[1] + b[1] * m1[0]) / determinant

    if 0. < alpha1 < 1. and 0. < alpha2 < 1.:
        # return b1 + alpha1 * m1
        return float(alpha1)

    return None

def is_on_right_side_of_turn(u, v, w):
    """
    Check whether `w` is on the outside of the turn from `u` to `v`.

    In detail, we draw `u` followed by `v` and `w` (the tails of `v` and
    `w` are at the head of `u`). This function returns `True` when `w`
    is more clockwise (closer to the right side of `u` when `u` is drawn
    upright) than `v`.

    Ties favor `v`.
    """
    if np.all(u == -w):
        return False

    # Normalize the inputs and flip `u`
    u = -u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    w = w / np.linalg.norm(w)

    # The idea is that we orthonormalize (`u` being the first element of
    # the basis; a quarter turn counterclockwise of that being the
    # second) and check coordinates using simple inequalities
    u_prime = np.array([-u[1], u[0]])
    v1 = v @ u
    v2 = v @ u_prime
    w1 = w @ u
    w2 = w @ u_prime
    return (
        (w2 >= 0. and (v2 < 0. or w1 > v1))
        or (w2 < 0. and v2 < 0. and w1 < v1)
    )

def compute_border(
    vertices: typing.Iterable[npt.NDArray[np.float64]],
    edges: typing.Iterable[typing.Tuple[int, int]]
) -> typing.List[typing.List[float]]:
    # First, deal with parts of the graph overlapping each other in
    # degenerate ways.

    # Remove duplicate vertices and trivial edges
    id_to_vertex = [(vertex[0], vertex[1]) for vertex in vertices]
    vertex_to_id = {vertex: id for id, vertex in reversed(list(enumerate(id_to_vertex)))}
    id_to_id = [vertex_to_id[vertex] for vertex in id_to_vertex]
    edges = [
        (new_id1, new_id2)
        for edge in edges
        for new_id1 in (id_to_id[edge[0]],)
        for new_id2 in (id_to_id[edge[1]],)
        if new_id1 != new_id2
    ]

    # TODO: Join edges that overlap? Is this necessary? It seems
    # unlikely

    # Points on the border are either at one of the input points or at
    # the intersection of two edges. The border of the graph is stored
    # as a collection of list of these points, where the points are
    # listed counterclockwise around each of the connected components.
    #
    # For naming's sake, the set intersection points is called
    # `intersections`. The polygons are stored as a dictionary so that
    # we can easily add/remove them from the collection for merging
    # purposes.
    intersections = list(vertices)
    polygons = {}
    polygon_count = 0

    # Our strategy is to incrementally add edges
    for u_index, v_index in edges:
        u = intersections[u_index]
        v = intersections[v_index]

        # First, find all places where the candidate edge intersects the
        # current set of polygons. Store this as a set of polygon ids
        # as well as a collection of values of alpha describing where
        # along our candidate segment the intersection occurs. These
        # alphas also correspond to specific edges
        polygon_ids_to_join = set()
        edges_to_alphas = {}
        for polygon_id, polygon in polygons.items():
            for p_index in polygon:
                if u_index == p_index or v_index == p_index:
                    polygon_ids_to_join.add(polygon_id)

            edges_polygon = list(itertools.islice(itertools.pairwise(itertools.cycle(polygon)), len(polygon)))
            for p_index, q_index in edges_polygon:
                p = intersections[p_index]
                q = intersections[q_index]

                if (q_index, p_index) in edges_to_alphas:
                    edges_to_alphas[(p_index, q_index)] = edges_to_alphas[(q_index, p_index)]
                    continue

                # Check whether uv intersects pq
                alpha = compute_segment_segment_intersection(u, v, p, q)
                if alpha is not None:
                    intersection = u + alpha * (v - u)
                    intersection_index = len(intersections)
                    intersections.append(intersection)
                    polygon_ids_to_join.add(polygon_id)
                    edges_to_alphas[(p_index, q_index)] = (alpha, intersection_index)

        # When there are no intersections, check whether the segment is
        # on the interior. If not, create a new polygon
        if not polygon_ids_to_join:
            uv_midpoint = (u + v) / 2
            # TODO: Avoid recreating this list of polygons every
            # iteration
            if not is_in_interior_of_polygons(uv_midpoint, [
                [intersections[intersection_index] for intersection_index in polygon]
                for polygon in polygons.values()
            ]):
                polygons[polygon_count] = [u_index, v_index]
                polygon_count += 1
            continue

        # Figure out all possible edges in the joined polygon. There
        # might be some extraneous ones
        edges_joined_polygon = collections.defaultdict(list)
        for polygon_id in polygon_ids_to_join:
            polygon = polygons[polygon_id]
            edges_polygon = list(itertools.islice(itertools.pairwise(itertools.cycle(polygon)), len(polygon)))
            for p_index, q_index in edges_polygon:
                if (p_index, q_index) in edges_to_alphas:
                    _, intersection_index = edges_to_alphas[p_index, q_index]
                    edges_joined_polygon[p_index].append(intersection_index)
                    edges_joined_polygon[intersection_index].append(q_index)
                else:
                    edges_joined_polygon[p_index].append(q_index)
        alphas = list(sorted(set(
            (alpha, intersection_index)
            for alpha, intersection_index in edges_to_alphas.values()
        )))
        for (_, intersection_index1), (_, intersection_index2) in itertools.pairwise(alphas):
            edges_joined_polygon[intersection_index1].append(intersection_index2)
            edges_joined_polygon[intersection_index2].append(intersection_index1)
        if alphas:
            intersection_index_first = alphas[0][1]
            edges_joined_polygon[u_index].append(intersection_index_first)
            edges_joined_polygon[intersection_index_first].append(u_index)

            intersection_index_last = alphas[-1][1]
            edges_joined_polygon[v_index].append(intersection_index_last)
            edges_joined_polygon[intersection_index_last].append(v_index)
        else:
            edges_joined_polygon[u_index].append(v_index)
            edges_joined_polygon[v_index].append(u_index)

        # Now just trace around the boundary counterclockwise

        # Start by identifying the leftmost point
        intersection_indices_to_join = set()
        for polygon_id in polygon_ids_to_join:
            polygon = polygons[polygon_id]
            for intersection_index in polygon:
                intersection_indices_to_join.add(intersection_index)
        for (_, intersection_index) in alphas:
            intersection_indices_to_join.add(intersection_index)
        intersection_indices_to_join = list(intersection_indices_to_join)
        index_leftmost = intersection_indices_to_join[0]
        for intersection_index in intersection_indices_to_join[1:]:
            if intersections[intersection_index][0] < intersections[index_leftmost][0]:
                index_leftmost = intersection_index

        # Then note that the connecting edge from there is the most
        # counterclockwise one connecting to that point starting from
        # pointing directly right
        neighbor_indices = edges_joined_polygon[index_leftmost]
        neighbor_index_best = 0
        neighbor_best = intersections[neighbor_indices[0]]
        u = np.array([1., 0.])
        for neighbor_index in range(1, len(neighbor_indices)):
            neighbor = intersections[neighbor_indices[neighbor_index]]
            v = neighbor_best - intersections[index_leftmost]
            w = neighbor - intersections[index_leftmost]

            if is_on_right_side_of_turn(u, v, w):
                neighbor_index_best = neighbor_index
                neighbor_best = neighbor

        index_previous = index_leftmost
        index_current = neighbor_indices[neighbor_index_best]
        polygon_joined = [index_previous, index_current]

        # Now iteratively walk around the joined polygon by repeatedly
        # selecting the smallest counterclockwise turn
        while True:
            neighbor_indices = edges_joined_polygon[index_current]

            # Check whether the previous vertex is a candidate for the
            # next one. If so, start our search one after that (to
            # avoid premature looping)
            if index_previous in neighbor_indices:
                neighbor_index_best = (neighbor_indices.index(index_previous) + 1) % len(neighbor_indices)
            else:
                neighbor_index_best = 0
            neighbor_best = intersections[neighbor_indices[neighbor_index_best]]

            # Search through the neighbors and find the "rightmost" turn
            for neighbor_index in itertools.islice(
                itertools.cycle(range(len(neighbor_indices))),
                neighbor_index_best + 1,
                neighbor_index_best + len(neighbor_indices)
            ):
                neighbor = intersections[neighbor_indices[neighbor_index]]
                u = intersections[index_current] - intersections[index_previous]
                v = neighbor_best - intersections[index_current]
                w = neighbor - intersections[index_current]

                if is_on_right_side_of_turn(u, v, w):
                    neighbor_index_best = neighbor_index
                    neighbor_best = neighbor

            # Check whether we've gotten back to the beginning
            index_next = neighbor_indices[neighbor_index_best]
            if index_current == polygon_joined[0] and index_next == polygon_joined[1]:
                polygon_joined.pop()
                break

            # Add the new polygon vertex
            index_previous = index_current
            index_current = neighbor_indices[neighbor_index_best]
            polygon_joined.append(index_current)

        # Replace the old polygons with their join
        for polygon_id in polygon_ids_to_join:
            del polygons[polygon_id]
        polygons[polygon_count] = polygon_joined
        polygon_count += 1

    return [
        [intersections[intersection_index] for intersection_index in polygon]
        for polygon in polygons.values()
    ]

def distance_to_border(
    point: npt.NDArray[np.float64],
    border: typing.List[typing.List[float]]
) -> float:
    """
    Compute the distance from a point to a graph's border.

    If the point lies inside the graph, the distance is 0.
    """
    if is_in_interior_of_polygons(point, border):
        return 0.

    return min([
        np.linalg.norm(point - project_to_line_segment(point, left, right))
        for polygon in border
        for left, right in itertools.islice(itertools.cycle(itertools.pairwise(polygon)), len(polygon))
    ])
