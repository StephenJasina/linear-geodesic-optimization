import numpy as np

def compute_convex_hull(points):
    '''
    An implementation of the Graham scan algorithm. As input, take a list of
    pairs. Returns a list of the indices of the vertices on the convex hull,
    oriented counter-clockwise.
    '''

    points = np.array(points)

    pivot_point_index = 0
    pivot_point = points[pivot_point_index]
    for index, point in enumerate(points):
        if point[1] < pivot_point[1] or (point[1] == pivot_point[1] and point[0] < pivot_point[0]):
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
        for index, point in enumerate(points)
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

            if np.cross(point_top - point_index, point_second_to_top - point_index) < 0:
                break

            del(convex_hull[-1])

        convex_hull.append(index)

    return convex_hull

def is_in_convex_hull(point, points, convex_hull=None):
    '''
    Return whether a point lies in a convex hull as computed by
    compute_convex_hull.
    '''

    point = np.array(point)

    if convex_hull is None:
        convex_hull = compute_convex_hull(points)

    left = points[convex_hull[-1]]
    right = points[convex_hull[0]]
    side = np.cross(point - left, right - left)

    for left_index, right_index in zip(convex_hull, convex_hull[1:]):
        left = points[left_index]
        right = points[right_index]
        if np.cross(point - left, right - left) * side < 0:
            return False
    return True
