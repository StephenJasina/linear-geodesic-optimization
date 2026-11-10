"""
Geodesic curvature of planar curves lifted onto a height-field surface.

A planar curve c lifts to the curve (c, h(c)) on the surface
z = h(x, y). Write t and n for the unit tangent and left normal of c,
k_p for its signed planar curvature, h_t = grad(h) . t,
h_n = grad(h) . n, and h_tt = t^T H t, where H is the Hessian of h.
The geodesic curvature of the lift is

    kappa_g = sqrt(1 + |grad(h)|^2)
              * (k_p + h_n h_tt / (1 + |grad(h)|^2))
              / (1 + h_t^2)^(3/2),

which follows from the Christoffel symbols
Gamma^m_ij = h_m h_ij / (1 + |grad(h)|^2) of the induced metric
I + grad(h) grad(h)^T. It is positive when the lift turns left as seen
from above. The lift is a geodesic exactly when kappa_g vanishes, so a
straight planar segment is a geodesic exactly when h_n h_tt = 0 along
it.

The derivatives of h come from the finite-difference operators of
`smooth_loss_tensor.grid_operators`, bilinearly interpolated between
grid vertices.
"""

import typing

import numpy as np
import numpy.typing as npt
import scipy.interpolate

from linear_geodesic_optimization.mesh.rectangle import Mesh
from linear_geodesic_optimization.optimization.smooth_loss_tensor \
    import grid_operators


def derivative_fields(
    mesh: Mesh,
    z: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Return the columns `[h_x, h_y, h_xx, h_yy, h_xy]` at every vertex.

    Boundary vertices have no finite-difference stencil, so their rows
    are NaN.
    """
    interior, operators = grid_operators(mesh)
    fields = np.full((mesh.get_topology().n_vertices, 5), np.nan)
    for column, operator in enumerate(operators):
        fields[interior, column] = operator @ z
    return fields


def interpolate(
    mesh: Mesh,
    fields: npt.NDArray[np.float64],
    points: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Bilinearly interpolate per-vertex `fields` at planar `points`.

    Points in a cell touching the boundary get NaN.
    """
    sides = mesh.get_width()
    spacing = mesh.get_scale() / (sides - 1)
    grid = np.asarray(points, dtype=np.float64)[:, :2] / spacing
    cell = np.clip(np.floor(grid).astype(np.int64), 0, sides - 2)
    fraction = grid - cell
    i, j = cell[:, 0], cell[:, 1]
    fx, fy = fraction[:, 0:1], fraction[:, 1:2]
    return (
        (1 - fx) * (1 - fy) * fields[i * sides + j]
        + fx * (1 - fy) * fields[(i + 1) * sides + j]
        + (1 - fx) * fy * fields[i * sides + j + 1]
        + fx * fy * fields[(i + 1) * sides + j + 1]
    )


def geodesic_curvature(
    derivatives: npt.NDArray[np.float64],
    tangents: npt.NDArray[np.float64],
    planar_curvatures: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Return kappa_g of the lifted curve at each sample.

    `derivatives` holds `[h_x, h_y, h_xx, h_yy, h_xy]` per sample,
    `tangents` the unit planar tangents, and `planar_curvatures` the
    signed planar curvatures.
    """
    h_x, h_y, h_xx, h_yy, h_xy = derivatives.T
    t_x, t_y = tangents.T
    h_t = h_x * t_x + h_y * t_y
    h_n = h_y * t_x - h_x * t_y
    h_tt = h_xx * t_x**2 + 2 * h_xy * t_x * t_y + h_yy * t_y**2
    g = 1 + h_x**2 + h_y**2
    return np.sqrt(g) * (planar_curvatures + h_n * h_tt / g) \
        / (1 + h_t**2)**1.5


def surface_speed(
    derivatives: npt.NDArray[np.float64],
    tangents: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Return the surface length per unit of planar length."""
    h_t = derivatives[:, 0] * tangents[:, 0] \
        + derivatives[:, 1] * tangents[:, 1]
    return np.sqrt(1 + h_t**2)


def turning_angle(
    derivatives: npt.NDArray[np.float64],
    tangents_in: npt.NDArray[np.float64],
    tangents_out: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Return the signed angle from the lift of each incoming direction to
    the lift of the outgoing direction, measured in the tangent plane.

    This is the geodesic turning that a corner of a lifted polyline
    concentrates at a single point. It is positive for left turns as
    seen from above.
    """
    gradients = derivatives[:, :2]

    def lift(tangents):
        lifted = np.column_stack(
            (tangents, (gradients * tangents).sum(axis=1))
        )
        return lifted / np.linalg.norm(lifted, axis=1, keepdims=True)

    lifted_in, lifted_out = lift(tangents_in), lift(tangents_out)
    normals = np.column_stack((-gradients, np.ones(len(gradients))))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    sine = (normals * np.cross(lifted_in, lifted_out)).sum(axis=1)
    cosine = (lifted_in * lifted_out).sum(axis=1)
    return np.arctan2(sine, cosine)


class Samples(typing.NamedTuple):
    """Midpoint samples of a planar curve."""

    positions: npt.NDArray[np.float64]
    tangents: npt.NDArray[np.float64]
    planar_curvatures: npt.NDArray[np.float64]
    planar_lengths: npt.NDArray[np.float64]
    segments: npt.NDArray[np.int64]
    """Index of the waypoint pair each sample lies between."""


def polyline_samples(
    waypoints: npt.NDArray[np.float64],
    spacing: float,
) -> Samples:
    """Sample each straight segment of a polyline about every `spacing`."""
    positions, tangents, lengths, segments = [], [], [], []
    for index, (a, b) in enumerate(zip(waypoints[:-1], waypoints[1:])):
        length = np.linalg.norm(b - a)
        if length == 0:
            continue
        n = max(int(np.ceil(length / spacing)), 1)
        fractions = (np.arange(n) + 0.5) / n
        positions.append(a + fractions[:, None] * (b - a))
        tangents.append(np.tile((b - a) / length, (n, 1)))
        lengths.append(np.full(n, length / n))
        segments.append(np.full(n, index))
    positions = np.concatenate(positions)
    return Samples(
        positions, np.concatenate(tangents), np.zeros(len(positions)),
        np.concatenate(lengths), np.concatenate(segments),
    )


def spline_samples(
    waypoints: npt.NDArray[np.float64],
    spacing: float,
) -> Samples:
    """
    Sample a natural cubic spline through the waypoints, parametrized
    by chord length, about every `spacing`.
    """
    if len(waypoints) == 2:
        return polyline_samples(waypoints, spacing)
    knots = np.concatenate(([0.], np.cumsum(
        np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    )))
    spline = scipy.interpolate.CubicSpline(
        knots, waypoints, bc_type='natural'
    )
    n = max(int(np.ceil(knots[-1] / spacing)), 1)
    u = (np.arange(n) + 0.5) * knots[-1] / n
    first, second = spline(u, 1), spline(u, 2)
    speed = np.linalg.norm(first, axis=1)
    cross = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    return Samples(
        spline(u), first / speed[:, None], cross / speed**3,
        speed * knots[-1] / n,
        np.searchsorted(knots, u, side='right') - 1,
    )


def route_curvature(
    mesh: Mesh,
    fields: npt.NDArray[np.float64],
    waypoints: npt.NDArray[np.float64],
    spacing: float,
    curve: typing.Literal['polyline', 'spline'] = 'polyline',
) -> typing.Dict[str, typing.Any]:
    """
    Measure how far the lift of a route is from a geodesic.

    The route is the polyline through `waypoints`, or a natural cubic
    spline through them when `curve` is `'spline'`. The returned
    dictionary has:
    * `length`: surface length of the lift
    * `turning`: total absolute geodesic turning in radians, the
      integral of |kappa_g| ds plus, for a polyline, the absolute
      turning angles at its corners. It is 0 for a geodesic.
    * `smooth_turning` and `corner_turning`: the two parts of `turning`
    * `planar_turning`: the same quantity for the flat plane
    * `bending`: the integral of kappa_g^2 ds (corners excluded)
    * `max_kappa_g`: the largest |kappa_g| along the curve
    * `samples` and `kappa_g`: the samples and their kappa_g
    """
    waypoints = np.asarray(waypoints, dtype=np.float64)[:, :2]
    # Drop repeated consecutive waypoints so every segment has a
    # direction
    waypoints = waypoints[np.concatenate((
        [True], np.linalg.norm(np.diff(waypoints, axis=0), axis=1) > 0
    ))]
    if len(waypoints) < 2:
        raise ValueError('A route needs two distinct waypoints')
    if curve == 'polyline':
        samples = polyline_samples(waypoints, spacing)
    elif curve == 'spline':
        samples = spline_samples(waypoints, spacing)
    else:
        raise ValueError(f'Unknown curve type "{curve}"')
    derivatives = interpolate(mesh, fields, samples.positions)
    kappa_g = geodesic_curvature(
        derivatives, samples.tangents, samples.planar_curvatures
    )
    surface_lengths = samples.planar_lengths \
        * surface_speed(derivatives, samples.tangents)

    corner_turning = 0.
    planar_turning = float(
        (np.abs(samples.planar_curvatures) * samples.planar_lengths).sum()
    )
    if curve == 'polyline' and len(waypoints) > 2:
        directions = np.diff(waypoints, axis=0)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        corner_derivatives = interpolate(mesh, fields, waypoints[1:-1])
        corner_turning = float(np.abs(turning_angle(
            corner_derivatives, directions[:-1], directions[1:]
        )).sum())
        planar_turning += float(np.abs(turning_angle(
            np.zeros_like(corner_derivatives), directions[:-1], directions[1:]
        )).sum())

    smooth_turning = float((np.abs(kappa_g) * surface_lengths).sum())
    return {
        'length': float(surface_lengths.sum()),
        'turning': smooth_turning + corner_turning,
        'smooth_turning': smooth_turning,
        'corner_turning': corner_turning,
        'planar_turning': planar_turning,
        'bending': float((kappa_g**2 * surface_lengths).sum()),
        'max_kappa_g': float(np.abs(kappa_g).max()),
        'samples': samples,
        'kappa_g': kappa_g,
    }
