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
