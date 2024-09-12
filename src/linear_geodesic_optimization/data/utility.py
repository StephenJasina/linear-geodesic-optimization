import typing

import numpy as np


def mercator(
    longitude: typing.Optional[np.float64] = None,
    latitude: typing.Optional[np.float64] = None
) -> typing.Union[
    np.float64,
    typing.Tuple[np.float64, np.float64]
]:
    '''
    Given a longitude in [-180, 180] and a latitude in [-90, 90], return
    an (x, y) pair representing the location on a Mercator projection.
    Assuming the latitude is no larger/smaller than +/- 85
    (approximately), the pair will lie in [-0.5, 0.5]^2.
    '''
    x = longitude / 360. if longitude is not None else None
    y = np.log(np.tan(np.pi / 4. + latitude * np.pi / 360.)) / (2. * np.pi) if latitude is not None else None

    if x is None:
        return y
    if y is None:
        return x
    return (x, y)

def inverse_mercator(
    x: typing.Optional[np.float64] = None,
    y: typing.Optional[np.float64] = None
) -> typing.Union[
    np.float64,
    typing.Tuple[np.float64, np.float64]
]:
    longitude = x * 360. if x is not None else None
    latitude = np.arctan(np.exp(y * 2. * np.pi)) * 360. / np.pi - 90. if y is not None else None

    if longitude is None:
        return latitude
    if latitude is None:
        return longitude
    return (longitude, latitude)

def get_sphere_point(latlong):
    """Convert spherical coordinates to Cartesian coordinates."""
    latlong = np.array(latlong) * np.pi / 180.
    return np.array([
        np.cos(latlong[1]) * np.cos(latlong[0]),
        np.sin(latlong[1]) * np.cos(latlong[0]),
        np.sin(latlong[0])
    ])

def get_latlong(p):
    """Convert Cartesian coordinates to spherical coordinates."""
    return (
        np.arcsin(p[2]) * 180. / np.pi,
        np.arctan2(p[1], p[0]) * 180. / np.pi
    )

def get_spherical_distance(a, b):
    """Find the distance on the unit sphere between two unit vectors."""
    # The value of (a @ b) is clamped between -1 and 1 to avoid issues
    # with floating point
    if np.all(a == b):
        return np.float64(0.)
    return np.arccos(min(max(a @ b, -1.), 1.))

def get_GCL(latlong_a, latlong_b):
    """Find the great circle latency between two points on Earth in ms."""
    # The following are in meters and meters per second
    circumference_earth = 40075016.68557849
    radius_earth = circumference_earth / (2 * np.pi)
    c = 299792458

    # Convert spherical coordinates to Cartesian coordinates
    p_a = get_sphere_point(latlong_a)
    p_b = get_sphere_point(latlong_b)

    # Special case for slightly improved numerical stability
    if np.all(p_a == p_b):
        return 0.

    # Compute the latency, which is the travel time at the rate of two
    # thirds the speed of light
    return 2 * 1000 * get_spherical_distance(p_a, p_b) * radius_earth \
        / (2. * c / 3.)
