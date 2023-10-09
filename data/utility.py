import numpy as np
from math import radians, sin, cos, sqrt, atan2, acos

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
    return np.arccos(min(max(a @ b, -1.), 1.))

# def get_GCD_latency(latlong_a, latlong_a):
#     """Find the great circle latency between two points on Earth."""
#     # The following are in meters and meters per second
#     circumference_earth = 40075016.68557849
#     radius_earth = 40075016.68557849 / (2 * np.pi)
#     c = 299792458
#
#     # Convert spherical coordinates to Cartesian coordinates
#     p_a = get_sphere_point(latlong_a)
#     p_b = get_sphere_point(latlong_b)
#
#     # Compute the latency, which is the travel time at the rate of two
#     # thirds the speed of light
#     return get_spherical_distance(p_a, p_b) * radius_earth / (2. * c / 3.)

def get_GCD_latency(latlong_a, latlong_b):
    """Find the great circle latency between two points on Earth."""
    # The following are in meters and meters per second
    circumference_earth = 40075016.68557849
    radius_earth = 40075016.68557849 / (2 * np.pi)
    c = 299792458

    # Convert spherical coordinates to Cartesian coordinates
    p_a = get_sphere_point(latlong_a)
    p_b = get_sphere_point(latlong_b)

    # Compute the latency, which is the travel time at the rate of two
    # thirds the speed of light
    distance_km = get_spherical_distance(p_a, p_b) * radius_earth / 1000
    speed_of_light_km_per_s = 299792.458
    # Speed in fiber optics is 2/3 the speed of light in vacuum
    speed_in_fiber_km_per_s = (2 / 3) * speed_of_light_km_per_s

    # Compute time taken (latency) in milliseconds (1 second = 1000 milliseconds)
    latency_ms = 5/4*(distance_km / speed_in_fiber_km_per_s) * 1000

    return latency_ms

