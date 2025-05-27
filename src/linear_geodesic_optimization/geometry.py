import numpy as np
import numpy.typing as npt

def distance_to_line_segment(
    u: npt.NDArray[np.float64],
    v: npt.NDArray[np.float64],
    r: npt.NDArray[np.float64],
):
    """
    Find the distance between a line segment `u`-`v` and a point `r`.

    This function operates assuming the l^2 norm.
    """
    ru = r - u
    rv = r - v
    uv = u - v

    if ru @ uv <= 0 and rv @ uv >= 0:
        # In this case, `r` is "between" `u` and `v`
        if uv @ uv == 0:
            # Need a special case when `u` and `v` coincide
            return np.sqrt(ru @ ru)
        squared_distance = rv @ rv - (uv @ rv)**2 / (uv @ uv)
        if squared_distance < 0:
            # Deal with floating point error
            return 0.
        return np.sqrt(squared_distance)
    else:
        # In this case, `r` is closest to one of the endpoints of the
        # segment
        return np.sqrt(min(ru @ ru, rv @ rv))

def is_in_triangle(
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    c: npt.NDArray[np.float64],
    p: npt.NDArray[np.float64]
):
    """
    Determine whether p is in the triangle abc.

    This function assumes inputs are 2-D coordinates.
    """
    def cross2d(x, y):
        return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]
    ab = cross2d(b - a, p - a)
    bc = cross2d(c - b, p - b)
    ca = cross2d(a - c, p - c)
    return (ab >= 0 and bc >= 0 and ca >= 0) or (ab <= 0 and bc <= 0 and ca <= 0)
