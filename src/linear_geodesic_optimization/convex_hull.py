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
    points = np.array(points)

    if convex_hull is None:
        convex_hull = compute_convex_hull(points)

    crosses = np.array([
        np.cross(point - points[left_index], points[right_index] - points[left_index])
        for left_index, right_index in zip(convex_hull, [*convex_hull[1:], convex_hull[0]])
    ])

    return np.all(crosses >= 0) or np.all(crosses <= 0)

def project_to_line(point, left, right):
    '''
    Project a point onto the line passing through left and right
    '''

    direction = right - left
    return left + (point - left) @ direction / (direction @ direction) * direction

def project_to_convex_hull(point, points, convex_hull=None):
    '''
    Return the distance to the convex hull as computed by compute_convex_hull.
    '''

    point = np.array(point)
    points = np.array(points)

    if convex_hull is None:
        convex_hull = compute_convex_hull(points)

    if is_in_convex_hull(point, points, convex_hull):
        return np.copy(point)

    projections = [
        project_to_line(point, points[left_index], points[right_index])
        for left_index, right_index in zip(convex_hull, [*convex_hull[1:], convex_hull[0]])
    ]
    projections = [
        projection
        for projection in projections
        if is_in_convex_hull(projection, points, convex_hull)
    ] + [points[index] for index in convex_hull]
    distances = [np.linalg.norm(point - projection) for projection in projections]
    return projections[np.argmin(distances)]
