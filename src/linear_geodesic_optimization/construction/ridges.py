"""
Construct mesh heights from raised ridges along the network's edges.

The ambient surface is the plane z = 0, and every feature lies above
it. Each network node gets a height from the throughput of its incident
edges, and each edge becomes a ridge whose crest runs between the
heights of its endpoints. In the edge frame (s along the edge, t across
it), the ridge is
    z_e(s, t) = c(s) * psi(t / w),
where psi is a compactly supported bump with psi(0) = 1 and w is the
ridge's half-width. On the crest (t = 0) the Gaussian curvature is
    K = c''(s) * c(s) * psi''(0) / w^2 / (1 + c'(s)^2)^2,
so the crest profile c solves the boundary value problem
    c'' = -kappa_e * w^2 * (1 + c'^2)^2 / (|psi''(0)| * c),
with c equal to the endpoint heights at either end. A negative kappa_e
makes the crest sag between its endpoints (a mountain pass on a raised
ridge), and a positive kappa_e makes it arch. Past the endpoints, the
ridge is capped by a round dome over the node.

The ridges are combined with a smooth maximum (a p-norm), so the surface
stays nonnegative and the higher ridge wins where ridges meet. Finally,
the field is smoothed with a Gaussian of a fixed physical width. This
rounds off the kinks where a sagging crest meets its node's dome, which
otherwise grow into curvature spikes as the mesh is refined (and stall
the L-BFGS polish). The boundary is left free: forcing it to 0 cuts
off ridges that reach past the mesh and creates the same kind of kink.
"""

import typing
import warnings

import numpy as np
import numpy.typing as npt
import scipy.integrate
import scipy.ndimage


# psi(u) = (1 - u^2)^3 on |u| < 1, which is C^2 with psi''(0) = -6
PSI_SECOND_DERIVATIVE = 6.


def psi(u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the cross profile (1 - u^2)^3, clipped to 0 for |u| >= 1."""
    return np.clip(1. - u**2, 0., None)**3


def transform_throughput(
    throughput: npt.NDArray[np.float64],
    transform: typing.Literal['linear', 'sqrt', 'log'],
) -> npt.NDArray[np.float64]:
    """Apply the throughput-to-height transform (before normalizing)."""
    throughput = np.maximum(np.asarray(throughput, dtype=np.float64), 0.)
    if transform == 'linear':
        return throughput
    if transform == 'sqrt':
        return np.sqrt(throughput)
    if transform == 'log':
        return np.log1p(throughput)
    raise ValueError(f'Unknown ridge_height_transform: {transform}')


def node_throughputs(
    n_nodes: int,
    network_edges: typing.Sequence[typing.Tuple[int, int]],
    throughputs: typing.Sequence[typing.Optional[float]],
    rule: typing.Literal['max', 'sum', 'mean'],
) -> npt.NDArray[np.float64]:
    """
    Return each node's throughput: the max, sum, or mean (`rule`) of
    its incident edges' throughputs (0 for isolated nodes).
    """
    if rule not in ('max', 'sum', 'mean'):
        raise ValueError(f'Unknown ridge_node_height: {rule}')
    incident = [[] for _ in range(n_nodes)]
    for (i, j), throughput in zip(network_edges, throughputs):
        throughput = 0. if throughput is None else float(throughput)
        incident[i].append(throughput)
        incident[j].append(throughput)
    reduce = {'max': np.max, 'sum': np.sum, 'mean': np.mean}[rule]
    return np.array([
        reduce(values) if values else 0.
        for values in incident
    ])


def node_heights(
    n_nodes: int,
    network_edges: typing.Sequence[typing.Tuple[int, int]],
    throughputs: typing.Sequence[typing.Optional[float]],
    rule: typing.Literal['max', 'sum', 'mean'],
    transform: typing.Literal['linear', 'sqrt', 'log'],
    height: float,
    base: float,
    reference: typing.Optional[float] = None,
) -> npt.NDArray[np.float64]:
    """
    Return each node's height.

    A node's throughput comes from `node_throughputs`. After
    `transform`, the values are normalized by the transformed
    `reference` throughput (clipped to [0, 1]), or by their own maximum
    when `reference` is None, so the heights are
    `height * (base + (1 - base) * normalized)`. A shared `reference`
    (e.g. the largest node throughput across an animation's snapshots)
    keeps heights comparable between snapshots. The `base` fraction
    keeps nodes with little or no throughput above the ambient plane.
    """
    values = transform_throughput(
        node_throughputs(n_nodes, network_edges, throughputs, rule), transform
    )
    if reference is None:
        maximum = np.max(values) if len(values) else 0.
    else:
        maximum = float(transform_throughput(np.array([reference]), transform)[0])
    normalized = np.clip(values / maximum, 0., 1.) if maximum > 0. else np.zeros_like(values)
    return height * (base + (1. - base) * normalized)


def crest_profile(
    length: float,
    height_a: float,
    height_b: float,
    curvature: typing.Optional[float],
    width: float,
    floor: float,
) -> typing.Tuple[typing.Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]], typing.Dict[str, typing.Any]]:
    """
    Return the crest height as a function of the arc length along the
    edge (in [0, `length`]), along with information about the solve.

    The profile solves the boundary value problem in the module
    docstring. If the solver fails, it falls back to the parabola with
    the same curvature at the mean endpoint height. Crest heights below
    `floor` are clamped to `floor`, which is recorded as `clamped`.
    """
    def linear(x):
        return height_a + (height_b - height_a) * np.asarray(x) / length

    if curvature is None or curvature == 0. or length == 0.:
        return (linear if length else (lambda x: np.full(np.shape(x), height_a))), {
            'solver': 'linear', 'clamped': False,
            'crest_min': float(min(height_a, height_b)),
            'crest_max': float(max(height_a, height_b)),
        }

    coefficient = -curvature * width**2 / PSI_SECOND_DERIVATIVE

    def rhs(x, y):
        c = np.maximum(y[0], floor)
        return np.vstack((y[1], coefficient * (1. + y[1]**2)**2 / c))

    def bc(ya, yb):
        return np.array([ya[0] - height_a, yb[0] - height_b])

    mesh = np.linspace(0., length, 33)
    guess = np.vstack((linear(mesh), np.full_like(mesh, (height_b - height_a) / length)))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        solution = scipy.integrate.solve_bvp(rhs, bc, mesh, guess, tol=1e-6, max_nodes=10000)

    samples = np.linspace(0., length, 257)
    if solution.status == 0 and np.all(np.isfinite(solution.sol(samples)[0])):
        solver = 'bvp'

        def profile(x):
            return solution.sol(np.asarray(x))[0]
    else:
        solver = 'parabola'
        mean_height = (height_a + height_b) / 2.
        second_derivative = coefficient / mean_height

        def profile(x):
            x = np.asarray(x)
            return linear(x) + 0.5 * second_derivative * x * (x - length)

    values = profile(samples)
    clamped = bool(np.min(values) < floor)

    def clamped_profile(x):
        return np.maximum(profile(x), floor)

    return clamped_profile, {
        'solver': solver,
        'clamped': clamped,
        'crest_min': float(max(np.min(values), floor)),
        'crest_max': float(np.max(values)),
    }


def ridge_field(
    points: npt.NDArray[np.float64],
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    profile: typing.Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    width: float,
) -> npt.NDArray[np.float64]:
    """
    Return the ridge z_e at `points`: the crest height at the projection
    onto the segment ab, times psi(distance to the segment / `width`).
    """
    direction = b - a
    length = np.linalg.norm(direction)
    if length:
        fraction = np.clip(((points - a) @ direction) / length**2, 0., 1.)
    else:
        fraction = np.zeros(len(points))
    distance = np.linalg.norm(points - (a + fraction[:, None] * direction), axis=1)
    near = distance < width
    z = np.zeros(len(points))
    z[near] = profile(fraction[near] * length) * psi(distance[near] / width)
    return z


def build_heights(
    points: npt.NDArray[np.float64],
    grid_shape: typing.Tuple[int, int],
    network_vertices: npt.NDArray[np.float64],
    network_edges: typing.Sequence[typing.Tuple[int, int]],
    network_curvatures: typing.Sequence[typing.Optional[float]],
    throughputs: typing.Sequence[typing.Optional[float]],
    width: float,
    height: float,
    base: float = 0.25,
    node_rule: typing.Literal['max', 'sum', 'mean'] = 'max',
    transform: typing.Literal['linear', 'sqrt', 'log'] = 'linear',
    blend_p: float = 8.,
    smoothing: float = 0.,
    reference: typing.Optional[float] = None,
) -> typing.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], typing.List[typing.Dict[str, typing.Any]]]:
    """
    Return `(z, node_heights, edges)`.

    `points` are the mesh vertices' (x, y) coordinates, ordered as a
    regular grid of shape `grid_shape` (as in `Mesh.get_coordinates`).
    `width` is the ridges' half-width, `height` the maximum node
    height, and `smoothing` the standard deviation of the final
    Gaussian smoothing (0 for none), all in mesh units. `reference` is
    the node throughput that maps to full height (see `node_heights`). `edges` holds one dictionary of diagnostics per edge.
    """
    network_vertices = np.asarray(network_vertices)[:, :2]
    heights = node_heights(
        len(network_vertices), network_edges, throughputs,
        node_rule, transform, height, base, reference
    )
    floor = 0.05 * base * height

    accumulated = np.zeros(len(points))
    edges = []
    for (i, j), curvature, throughput in zip(network_edges, network_curvatures, throughputs):
        a, b = network_vertices[i], network_vertices[j]
        length = float(np.linalg.norm(b - a))
        profile, info = crest_profile(
            length, heights[i], heights[j],
            None if curvature is None else float(curvature),
            width, floor
        )
        z_e = ridge_field(points, a, b, profile, width)
        accumulated += (z_e / height)**blend_p
        edges.append({
            'edge': [int(i), int(j)],
            'curvature': None if curvature is None else float(curvature),
            'throughput': None if throughput is None else float(throughput),
            'heights': [float(heights[i]), float(heights[j])],
            **info,
        })

    z = height * accumulated**(1. / blend_p)
    if smoothing > 0.:
        cell = np.linalg.norm(points[1] - points[0])
        z = scipy.ndimage.gaussian_filter(
            z.reshape(grid_shape), smoothing / cell, mode='nearest'
        ).ravel()
    return z, heights, edges
