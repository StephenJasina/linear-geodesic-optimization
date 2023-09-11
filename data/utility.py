import numpy as np

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
    return np.arccos(min(max(a @ b, -1.), 1.))

def get_GCD_latency(latlong_a, latlong_b):
    """Find the great circle distance between two points on Earth."""
    radius_earth = 40075016.68557849 / (2 * np.pi)
    c = 299792458

    # Convert spherical coordinates to Cartesian coordinates
    p_a = get_sphere_point(latlong_a)
    p_b = get_sphere_point(latlong_b)

    # Note that the dot product is clamped to [-1, 1] to deal with
    # floating point error
    return get_spherical_distance(p_a, p_b) * radius_earth / (2. * c / 3.)
