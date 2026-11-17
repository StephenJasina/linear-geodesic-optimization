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
import scipy.sparse

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


def bilinear_operator(
    mesh: Mesh,
    positions: npt.NDArray[np.float64],
) -> typing.Tuple[npt.NDArray[np.bool_], scipy.sparse.csr_array]:
    """
    Return `(keep, A)`, where `A` is the sparse matrix that bilinearly
    interpolates per-vertex values at the planar `positions`.

    Positions in cells touching the boundary are left out (they have no
    finite-difference stencil in `directional_operators`); `keep` marks
    the positions that remain, which index the rows of `A`.
    """
    positions = np.asarray(positions, dtype=np.float64)[:, :2]
    sides = mesh.get_width()
    grid = positions / (mesh.get_scale() / (sides - 1))
    cell = np.floor(grid).astype(np.int64)
    keep = np.all((cell >= 1) & (cell <= sides - 3), axis=1)
    grid, cell = grid[keep], cell[keep]
    fx, fy = (grid - cell).T
    i, j = cell.T
    n = len(grid)
    rows = np.tile(np.arange(n), 4)
    columns = np.concatenate((
        i * sides + j, (i + 1) * sides + j,
        i * sides + j + 1, (i + 1) * sides + j + 1,
    ))
    values = np.concatenate((
        (1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy,
    ))
    return keep, scipy.sparse.csr_array(
        (values, (rows, columns)), shape=(n, mesh.get_topology().n_vertices)
    )


def directional_operators(
    mesh: Mesh,
    positions: npt.NDArray[np.float64],
    tangents: npt.NDArray[np.float64],
) -> typing.Tuple[npt.NDArray[np.bool_], typing.Tuple[scipy.sparse.csr_array, ...]]:
    """
    Return `(keep, (A_t, A_n, A_tt))`, the sparse matrices taking the
    heights to `h_t`, `h_n` and `h_tt` at the planar `positions` with
    unit `tangents` (bilinearly interpolating the finite differences of
    `smooth_loss_tensor.grid_operators`).

    Positions in cells touching the boundary have no finite-difference
    stencil, so they are left out; `keep` marks the positions that
    remain, which index the matrices' rows.
    """
    tangents = np.asarray(tangents, dtype=np.float64)
    keep, interpolation = bilinear_operator(mesh, positions)
    tangents = tangents[keep]
    n_vertices = mesh.get_topology().n_vertices
    interior, operators = grid_operators(mesh)
    embed = scipy.sparse.csr_array(
        (np.ones(len(interior)), (interior, np.arange(len(interior)))),
        shape=(n_vertices, len(interior)),
    )
    d_x, d_y, d_xx, d_yy, d_xy = (
        interpolation @ embed @ operator for operator in operators
    )

    t_x = scipy.sparse.diags_array(tangents[:, 0])
    t_y = scipy.sparse.diags_array(tangents[:, 1])
    return keep, tuple(scipy.sparse.csr_array(operator) for operator in (
        t_x @ d_x + t_y @ d_y,
        t_x @ d_y - t_y @ d_x,
        t_x @ t_x @ d_xx + 2 * t_x @ t_y @ d_xy + t_y @ t_y @ d_yy,
    ))


class CurveBending:
    """
    Weighted bending energy of the lifts of planar curves, used as a
    loss that makes network routes geodesics.

    Since `1 + |grad(h)|^2 = 1 + h_t^2 + h_n^2`, the integrand
    `kappa_g^2 ds` of the bending energy, per unit of planar length, is

        f = (k_p g + h_n h_tt)^2 / (g (1 + h_t^2)^(5/2)),

    where `g = 1 + h_t^2 + h_n^2` and `k_p` is the planar curvature.
    At fixed samples, `h_t`, `h_n` and `h_tt` are linear in the
    heights, so they are precomputed as the sparse matrices
    `operators = (A_t, A_n, A_tt)`. The loss is `sum_i w_i f_i`, where
    `w_i` is the sample's planar length times its curve's weight, and
    the curve weights are normalized to sum to 1. A curve then
    contributes the integral of kappa_g^2 over its lift, which is 0
    exactly when the lift is a geodesic.

    Each curve is the polyline through its waypoints or, with `curve`
    `'spline'`, the natural cubic spline through them (see
    `polyline_samples` and `spline_samples`). The corners of a polyline
    are not counted, so polylines are best used for straight segments.

    Samples in cells touching the boundary have no finite-difference
    stencil and are dropped.
    """

    def __init__(
        self,
        mesh: Mesh,
        curves: typing.Sequence[npt.NDArray[np.float64]],
        weights: npt.NDArray[np.float64],
        spacing: float,
        curve: typing.Literal['polyline', 'spline'] = 'polyline',
    ):
        weights = np.asarray(weights, dtype=np.float64)
        weights = weights / weights.sum()
        sample = {'polyline': polyline_samples, 'spline': spline_samples}[curve]

        positions, tangents, curvatures, sample_weights = [], [], [], []
        for waypoints, weight in zip(curves, weights):
            waypoints = np.asarray(waypoints, dtype=np.float64)[:, :2]
            waypoints = waypoints[np.concatenate((
                [True],
                np.linalg.norm(np.diff(waypoints, axis=0), axis=1) > 0,
            ))]
            if len(waypoints) < 2:
                continue
            samples = sample(waypoints, spacing)
            positions.append(samples.positions)
            tangents.append(samples.tangents)
            curvatures.append(samples.planar_curvatures)
            sample_weights.append(weight * samples.planar_lengths)
        positions = np.concatenate(positions)
        tangents = np.concatenate(tangents)
        curvatures = np.concatenate(curvatures)
        sample_weights = np.concatenate(sample_weights)

        keep, self.operators = directional_operators(mesh, positions, tangents)
        self.n_dropped = int((~keep).sum())
        positions, tangents, curvatures, sample_weights = (
            array[keep] for array in
            (positions, tangents, curvatures, sample_weights)
        )
        self.positions = positions
        self.tangents = tangents
        self.planar_curvatures = curvatures
        self.sample_weights = sample_weights

    def evaluate(
        self, z: npt.NDArray[np.float64]
    ) -> typing.Tuple[float, npt.NDArray[np.float64]]:
        """Return the loss and its gradient with respect to `z`."""
        a_t, a_n, a_tt = self.operators
        h_t, h_n, h_tt = a_t @ z, a_n @ z, a_tt @ z
        k = self.planar_curvatures
        q = 1 + h_t**2
        g = q + h_n**2
        numerator = k * g + h_n * h_tt
        denominator = g * q**2.5
        f = numerator**2 / denominator
        w = self.sample_weights
        d_t = 4 * numerator * k * h_t / denominator \
            - f * (2 * h_t / g + 5 * h_t / q)
        d_n = 2 * numerator * (2 * k * h_n + h_tt) / denominator \
            - f * 2 * h_n / g
        d_tt = 2 * numerator * h_n / denominator
        gradient = a_t.T @ (w * d_t) + a_n.T @ (w * d_n) \
            + a_tt.T @ (w * d_tt)
        return float(w @ f), gradient


def _label_indices(network):
    graph_data, vertex_data, _ = network
    label_to_index = {
        str(label): index for index, label in enumerate(graph_data['labels'])
    }
    for index, elements in enumerate(vertex_data.get('elements', []) or []):
        for element in elements or []:
            label_to_index.setdefault(str(element), index)
    return label_to_index


def route_curves(
    network, routes, traffic,
    kind: typing.Literal['route', 'link'] = 'route',
) -> typing.Tuple[typing.List[typing.Tuple[int, ...]], npt.NDArray[np.float64]]:
    """
    Return `(curves, weights)`: tuples of indices of the network's
    vertices, each weighted by the total traffic through it.

    With `kind` `'route'`, each curve is a whole route, with a route
    and its reverse counted as one curve. With `kind` `'link'`, each
    curve is a link `(i, j)` used by some route, in either direction.
    Route waypoints are vertex labels, or labels of elements merged into
    a vertex by clustering. Repeated consecutive waypoints are merged,
    and curves with no traffic or a single vertex are left out.
    """
    label_to_index = _label_indices(network)
    totals = {}
    for route, volume in zip(routes, traffic):
        indices = [label_to_index[str(label)] for label in route]
        indices = [
            index for position, index in enumerate(indices)
            if position == 0 or index != indices[position - 1]
        ]
        if kind == 'route':
            keys = [min(tuple(indices), tuple(indices[::-1]))]
        elif kind == 'link':
            keys = [
                (min(i, j), max(i, j))
                for i, j in zip(indices[:-1], indices[1:])
            ]
        else:
            raise ValueError(f'Unknown curve kind "{kind}"')
        for key in keys:
            if len(key) >= 2:
                totals[key] = totals.get(key, 0.) + float(volume)
    curves = [key for key, total in totals.items() if total > 0]
    return curves, np.array([totals[key] for key in curves])
