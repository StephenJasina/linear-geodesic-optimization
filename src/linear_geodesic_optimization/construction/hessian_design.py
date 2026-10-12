"""
Construct mesh heights directly from network Ricci curvatures.

For a height field f with small slopes, the Gaussian curvature is
approximately det(Hess f). Prescribing det(Hess f) is nonlinear, but
prescribing the full Hessian is a linear least squares problem in f.

Each network edge e with Ollivier-Ricci curvature kappa_e is turned into
a target Hessian on the mesh vertices near it. In the frame (u, v),
where u runs along the edge and v across it, the target is
    T_e = alpha * u u^T + beta * v v^T,    alpha * beta = kappa_e.
A negative kappa_e gives a mountain pass (curving up along the edge and
falling away across it), and a positive kappa_e gives a dome. The
heights are then the least squares fit of the discrete Hessian to these
targets, with a thin-plate regularizer elsewhere and zero heights on
the boundary of the mesh. Optionally, a few local/global rounds then
let each target rotate and change aspect ratio (keeping its
determinant) so that targets meeting at a node become compatible.
"""

import typing

import numpy as np
import numpy.typing as npt
import scipy.sparse
import scipy.sparse.linalg


def tube_members(
    points: npt.NDArray[np.float64],
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    radius: float,
) -> npt.NDArray[np.int64]:
    """
    Return the indices of `points` within `radius` of the segment ab.

    This is the same membership test used by the torch backend's
    curvature loss.
    """
    direction = b - a
    length_squared = direction @ direction
    if length_squared:
        fraction = np.clip(((points - a) @ direction) / length_squared, 0., 1.)
    else:
        fraction = np.zeros(len(points))
    distance_squared = np.sum(
        (points - (a + fraction[:, None] * direction))**2, axis=1
    )
    return np.flatnonzero(distance_squared < radius**2)


def edge_target(
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    curvature: float,
    anisotropy: float = 1.,
    orientation: typing.Literal['hill', 'basin'] = 'hill',
) -> npt.NDArray[np.float64]:
    """
    Return the target Hessian (T_xx, T_yy, T_xy) for the edge ab.

    The determinant of the target is `curvature`. The `anisotropy`
    parameter is the ratio of the along-edge to the across-edge
    principal curvature magnitudes. With orientation `'hill'`, positive
    curvature gives a dome, and negative curvature gives a pass that
    rises along the edge. Orientation `'basin'` negates everything,
    which leaves the curvature unchanged.
    """
    u = b - a
    norm = np.linalg.norm(u)
    u = u / norm if norm else np.array([1., 0.])
    v = np.array([-u[1], u[0]])

    root = np.sqrt(abs(curvature))
    if curvature < 0.:
        alpha, beta = anisotropy * root, -root / anisotropy
    else:
        alpha, beta = -anisotropy * root, -root / anisotropy
    if orientation == 'basin':
        alpha, beta = -alpha, -beta
    elif orientation != 'hill':
        raise ValueError(f'Unknown orientation "{orientation}"')

    return np.array([
        alpha * u[0] * u[0] + beta * v[0] * v[0],
        alpha * u[1] * u[1] + beta * v[1] * v[1],
        alpha * u[0] * u[1] + beta * v[0] * v[1],
    ])


def hessian_operators(
    sides: int,
    scale: float,
) -> typing.Tuple[scipy.sparse.csr_array, scipy.sparse.csr_array,
                  scipy.sparse.csr_array, npt.NDArray[np.bool_]]:
    """
    Return finite-difference Hessian operators on a square grid.

    The grid matches `RectangleMesh`: vertex `i * sides + j` sits at
    `(i, j) * scale / (sides - 1)`, and each cell is cut along its
    diagonal from `(i, j)` to `(i + 1, j + 1)`. The returned operators
    `(Dxx, Dyy, Dxy)` map heights on all vertices to Hessian entries on
    interior vertices. The boolean mask selecting the interior vertices
    is returned as well.

    Dxx and Dyy are the usual second differences. Dxy is computed from
    the second difference along the mesh diagonal, which is
    f_xx + 2 f_xy + f_yy. Unlike the central-difference stencil, this
    only uses mesh neighbors, which keeps the fitted Hessian consistent
    with the angle-defect curvature computed from vertex one-rings. All
    three operators are exact on quadratics.
    """
    h = scale / (sides - 1)
    interior_1d = np.zeros(sides, dtype=bool)
    interior_1d[1:-1] = True
    interior = np.outer(interior_1d, interior_1d).ravel()

    d2 = scipy.sparse.diags(
        [1., -2., 1.], [-1, 0, 1], shape=(sides, sides)
    ) / h**2
    identity = scipy.sparse.identity(sides)
    # Interior rows never wrap around, since their diagonal neighbors
    # are on the grid
    d_diagonal = scipy.sparse.diags(
        [1., -2., 1.], [-(sides + 1), 0, sides + 1],
        shape=(sides * sides, sides * sides)
    ) / h**2

    dxx = scipy.sparse.csr_array(scipy.sparse.kron(d2, identity))[interior]
    dyy = scipy.sparse.csr_array(scipy.sparse.kron(identity, d2))[interior]
    d_diagonal = scipy.sparse.csr_array(d_diagonal)[interior]
    return dxx, dyy, (d_diagonal - dxx - dyy) / 2, interior


def gradient_operators(
    sides: int,
    scale: float,
) -> typing.Tuple[scipy.sparse.csr_array, scipy.sparse.csr_array]:
    """
    Return central-difference gradient operators on a square grid.

    The grid is the same as in `hessian_operators`. The returned
    operators `(Dx, Dy)` map heights on all vertices to gradient
    entries on interior vertices.
    """
    h = scale / (sides - 1)
    interior_1d = np.zeros(sides, dtype=bool)
    interior_1d[1:-1] = True
    interior = np.outer(interior_1d, interior_1d).ravel()

    d1 = scipy.sparse.diags(
        [-1., 1.], [-1, 1], shape=(sides, sides)
    ) / (2. * h)
    identity = scipy.sparse.identity(sides)

    dx = scipy.sparse.csr_array(scipy.sparse.kron(d1, identity))[interior]
    dy = scipy.sparse.csr_array(scipy.sparse.kron(identity, d1))[interior]
    return dx, dy


def cotangent_weights(
    mesh,
    z: npt.NDArray[np.float64],
) -> typing.Tuple[npt.NDArray[np.int64], npt.NDArray[np.float64], float]:
    """
    Return the cotangent Laplacian weights of the mesh at heights z.

    The result is `(edges, weights, total_area)`, where `edges` lists
    the edges between two interior vertices as rows `(u, v)` with
    `u < v`, `weights` holds (cot alpha + cot beta) / 2 for each of
    them, and `total_area` is the surface area. These match the legacy
    laplacian's `LC_interior_edges` and `A`.
    """
    topology = mesh.get_topology()
    faces = np.array([
        [vertex.index for vertex in face.vertices()]
        for face in topology.faces()
    ])
    boundary = np.array([
        vertex.is_on_boundary() for vertex in topology.vertices()
    ])
    points = np.concatenate(
        (mesh.get_coordinates()[:, :2], z[:, None]), axis=1
    )
    total_area = 0.
    keys = []
    values = []
    for corner in range(3):
        w = faces[:, corner]
        a = faces[:, (corner + 1) % 3]
        b = faces[:, (corner + 2) % 3]
        ea, eb = points[a] - points[w], points[b] - points[w]
        double_area = np.linalg.norm(np.cross(ea, eb), axis=1)
        if corner == 0:
            total_area = np.sum(double_area) / 2.
        keep = ~boundary[a] & ~boundary[b]
        keys.append(np.sort(np.stack([a, b], axis=1)[keep], axis=1))
        values.append((np.sum(ea * eb, axis=1) / double_area / 2.)[keep])
    edges, inverse = np.unique(
        np.concatenate(keys), axis=0, return_inverse=True
    )
    weights = np.zeros(len(edges))
    np.add.at(weights, inverse.ravel(), np.concatenate(values))
    return edges, weights, float(total_area)


def smoothness_operator(
    mesh,
    z: typing.Optional[npt.NDArray[np.float64]] = None,
    clamp_negative_weights: bool = True,
) -> scipy.sparse.csr_array:
    """
    Return the matrix Q of the linearized tensor smoothness loss.

    The tensor smoothness loss (see
    `linear_geodesic_optimization.optimization.smooth_loss_tensor`) is
        A_total * sum_{interior edges uv} w_uv ||S_u - R_uv S_v R_uv^T||^2,
    where the shape operator S = P (Hess f / W) P^T depends on the
    heights through Hess f (linearly) and through the slopes (in P and
    W). Freezing the slopes, the normals' transports R, the cotangent
    weights w, and the total area at the heights `z` leaves a quadratic
    form f^T Q f in the interior heights f (boundary heights are 0). At
    f = z (with zero boundary), it equals the tensor smoothness loss
    exactly. The cotangent weights of the grid's diagonal edges turn
    negative as soon as the surface bends, so by default negative
    weights are clamped to 0, which keeps Q positive semidefinite.

    With z = None (a flat mesh), the slopes vanish and the transports
    are the identity, so this is the third-order energy
        A_total * sum_{axis edges uv} ||Hess f(u) - Hess f(v)||_F^2,
    a discrete version of the integral of ||grad Hess f||^2.
    """
    # Imported here to avoid a circular import
    from linear_geodesic_optimization.optimization.smooth_loss_tensor \
        import frames, transports

    sides = mesh.get_width()
    scale = mesh.get_scale()
    if z is None:
        z = np.zeros(sides * sides)
    dxx, dyy, dxy, interior = hessian_operators(sides, scale)
    dx, dy = gradient_operators(sides, scale)
    interior_indices = np.flatnonzero(interior)
    n_interior = len(interior_indices)
    row_of = np.full(sides * sides, -1)
    row_of[interior_indices] = np.arange(n_interior)

    p, w, normals = frames(np.stack([dx @ z, dy @ z], axis=1))
    # Shape operator contribution of each Hessian channel, as rows
    # (f_xx, f_yy, f_xy) of 3x3 matrices
    basis = np.zeros((3, 2, 2))
    basis[0, 0, 0] = basis[1, 1, 1] = 1.
    basis[2, 0, 1] = basis[2, 1, 0] = 1.
    channels = np.einsum(
        'vik,ckl,vjl->vcij', p, basis, p
    ) / w[:, None, None, None]

    edges, weights, total_area = cotangent_weights(mesh, z)
    u, v = row_of[edges[:, 0]], row_of[edges[:, 1]]
    r, _, _ = transports(normals[u], normals[v])
    channels_v = np.einsum(
        'eik,eckl,ejl->ecij', r, channels[v], r
    )

    # Rows are (edge, matrix entry); columns are (channel, interior row)
    n_edges = len(edges)
    entry_rows = np.arange(n_edges * 9).reshape(n_edges, 1, 9)
    entry_rows = np.broadcast_to(entry_rows, (n_edges, 3, 9))
    channel_columns = np.arange(3)[None, :, None] * n_interior
    rows = np.concatenate([entry_rows.ravel(), entry_rows.ravel()])
    columns = np.concatenate([
        np.broadcast_to(channel_columns + u[:, None, None], (n_edges, 3, 9)).ravel(),
        np.broadcast_to(channel_columns + v[:, None, None], (n_edges, 3, 9)).ravel(),
    ])
    data = np.concatenate([
        channels[u].reshape(n_edges, 3, 9).ravel(),
        -channels_v.reshape(n_edges, 3, 9).ravel(),
    ])
    differences = scipy.sparse.csr_array(
        (data, (rows, columns)), shape=(n_edges * 9, 3 * n_interior)
    )
    hessians = scipy.sparse.vstack(
        [d[:, interior] for d in (dxx, dyy, dxy)], format='csr'
    )
    operator = differences @ hessians
    edge_weights = scipy.sparse.diags_array(
        total_area * np.repeat(
            np.maximum(weights, 0.) if clamp_negative_weights else weights, 9
        )
    )
    return scipy.sparse.csr_array(operator.T @ edge_weights @ operator)


def edge_weights(
    network_vertices: npt.NDArray[np.float64],
    network_edges: typing.List[typing.Tuple[int, int]],
) -> npt.NDArray[np.float64]:
    """Return edge weights proportional to length, as in the loss."""
    lengths = np.array([
        np.linalg.norm(network_vertices[i, :2] - network_vertices[j, :2])
        for i, j in network_edges
    ])
    return lengths / lengths.sum()


def project_to_determinant(
    hessians: npt.NDArray[np.float64],
    curvatures: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Return the nearest symmetric matrices with prescribed determinants.

    Both the input and output Hessians are given as rows (H_xx, H_yy,
    H_xy), and nearness is measured in the Frobenius norm. The nearest
    matrix shares eigenvectors with the input, so this reduces to
    finding the nearest point (mu_1, mu_2) on the hyperbola
    mu_1 mu_2 = kappa to the eigenvalues (lambda_1, lambda_2). Writing
    the stationarity conditions as mu_1 - lambda_1 = t mu_2 and
    mu_2 - lambda_2 = t mu_1 gives a quartic in t.
    """
    matrices = np.empty((len(hessians), 2, 2))
    matrices[:, 0, 0] = hessians[:, 0]
    matrices[:, 1, 1] = hessians[:, 1]
    matrices[:, 0, 1] = matrices[:, 1, 0] = hessians[:, 2]
    eigenvalues, eigenvectors = np.linalg.eigh(matrices)
    lambda_1, lambda_2 = eigenvalues[:, 0], eigenvalues[:, 1]
    kappa = np.asarray(curvatures, dtype=np.float64)

    # Default (also used for kappa = 0): zero the smaller eigenvalue
    small_1 = np.abs(lambda_1) < np.abs(lambda_2)
    mu_1 = np.where(small_1, 0., lambda_1)
    mu_2 = np.where(small_1, lambda_2, 0.)

    nonzero = kappa != 0.
    if np.any(nonzero):
        l1, l2, k = lambda_1[nonzero], lambda_2[nonzero], kappa[nonzero]
        # -k t^4 + (l1 l2 + 2k) t^2 + (l1^2 + l2^2) t + (l1 l2 - k) = 0,
        # solved through the eigenvalues of the companion matrices
        coefficients = np.stack([
            np.zeros_like(k), (l1 * l2 + 2 * k), (l1**2 + l2**2), (l1 * l2 - k)
        ], axis=1) / -k[:, None]
        companion = np.zeros((len(k), 4, 4))
        companion[:, 1:, :-1] = np.eye(3)
        companion[:, :, -1] = -coefficients[:, ::-1]
        roots = np.linalg.eigvals(companion)

        t = roots.real
        valid = (np.abs(roots.imag) < 1e-9) & (np.abs(np.abs(t) - 1.) > 1e-12)
        t = np.where(valid, t, 0.)
        denominator = np.where(valid, 1. - t**2, 1.)
        candidate_1 = (l1[:, None] + t * l2[:, None]) / denominator
        candidate_2 = (l2[:, None] + t * l1[:, None]) / denominator
        distance = np.where(
            valid,
            (candidate_1 - l1[:, None])**2 + (candidate_2 - l2[:, None])**2,
            np.inf
        )
        best = np.argmin(distance, axis=1)
        found = np.isfinite(distance[np.arange(len(k)), best])
        # A real stationary point always exists; fall back to an
        # isotropic matrix only in case of numerical trouble
        root = np.sqrt(np.abs(k))
        mu_1[nonzero] = np.where(
            found, candidate_1[np.arange(len(k)), best], root
        )
        mu_2[nonzero] = np.where(
            found, candidate_2[np.arange(len(k)), best], np.sign(k) * root
        )

    projected = eigenvectors @ (
        np.stack([mu_1, mu_2], axis=1)[:, :, None]
        * np.swapaxes(eigenvectors, 1, 2)
    )
    return np.stack(
        [projected[:, 0, 0], projected[:, 1, 1], projected[:, 0, 1]], axis=1
    )


def solve_heights(
    mesh,
    network_vertices: npt.NDArray[np.float64],
    network_edges: typing.List[typing.Tuple[int, int]],
    network_curvatures: typing.List[typing.Optional[float]],
    tube_radius: float,
    anisotropy: float = 1.,
    orientation: typing.Literal['hill', 'basin'] = 'hill',
    regularization: float = 1e-3,
    iterations: int = 0,
    slope_correction: bool = False,
    smoothness: float = 0.,
) -> npt.NDArray[np.float64]:
    """
    Fit heights whose Hessian matches the edge targets.

    Minimize
        sum_e sum_{v in tube(e)} w_e / |tube(e)| * ||H f(v) - T_{e,v}||_F^2
            + regularization * mean_v ||H f(v)||_F^2
    over heights f that vanish on the boundary of the mesh, where the
    inner sums run over interior mesh vertices. The weights w_e sum to
    1, so `regularization` is relative to the total target weight.

    Initially, T_{e,v} = T_e is the designed target from `edge_target`.
    The designed targets of edges meeting at a node generally cannot
    all be met at once, so each of the `iterations` subsequent rounds
    replaces T_{e,v} with the matrix of determinant kappa_e nearest to
    the current H f(v) and solves again. This alternation (as in
    as-rigid-as-possible deformation) monotonically decreases the
    objective over both f and the targets, and only lets the targets
    rotate and change aspect ratio while keeping their determinants.

    The fit treats det(H f) as the Gaussian curvature, which only holds
    for small slopes; in general the curvature is
    det(H f) / (1 + |grad f|^2)^2. With `slope_correction`, each round
    instead projects onto the determinant kappa_e (1 + |grad f(v)|^2)^2,
    using the slopes of the previous round's heights, so that steep
    surfaces still meet their curvature targets. This has no effect
    when `iterations` is 0.

    With positive `smoothness`, the objective also includes
    `smoothness` times the tensor smoothness loss, linearized as in
    `smoothness_operator`. Without `slope_correction`, it is linearized
    about the flat mesh, giving a third-order (integral of
    ||grad Hess f||^2) penalty that, unlike `regularization`, does not
    penalize uniform curvature. With `slope_correction`, each round
    instead freezes the slopes of the previous round's heights, so that
    the penalty tracks the true loss on steep surfaces.

    Return the heights of all mesh vertices, in mesh order.
    """
    sides = mesh.get_width()
    dxx, dyy, dxy, interior = hessian_operators(sides, mesh.get_scale())
    interior_indices = np.flatnonzero(interior)
    n_interior = len(interior_indices)
    # Map from mesh vertex index to row of the Hessian operators
    row_of = np.full(sides * sides, -1)
    row_of[interior_indices] = np.arange(n_interior)

    # Each (tube vertex, edge) pair contributes one target. Its row,
    # weight, and curvature are fixed; only its target changes between
    # iterations.
    weight = np.full(n_interior, regularization / n_interior)
    pair_rows = []
    pair_weights = []
    pair_curvatures = []
    pair_targets = []
    points = mesh.get_coordinates()[:, :2]
    for (i, j), curvature, w in zip(
        network_edges, network_curvatures,
        edge_weights(network_vertices, network_edges)
    ):
        if curvature is None:
            continue
        a, b = network_vertices[i, :2], network_vertices[j, :2]
        rows = row_of[tube_members(points, a, b, tube_radius)]
        rows = rows[rows >= 0]
        if len(rows) == 0:
            continue
        w_v = w / len(rows)
        weight[rows] += w_v
        pair_rows.append(rows)
        pair_weights.append(np.full(len(rows), w_v))
        pair_curvatures.append(np.full(len(rows), curvature))
        pair_targets.append(np.tile(
            edge_target(a, b, curvature, anisotropy, orientation),
            (len(rows), 1)
        ))

    # Restrict to interior unknowns (boundary heights are 0). The xy
    # component appears twice in the Frobenius norm. Writing c_v for
    # the total weight at v and t_v for its weighted target sum, the
    # objective is (up to a constant) sum_v c_v ||H f(v) - t_v / c_v||^2,
    # so the system matrix does not depend on the targets.
    operators = [d[:, interior] for d in (dxx, dyy, dxy)]
    if slope_correction:
        gradients = [
            d[:, interior] for d in gradient_operators(sides, mesh.get_scale())
        ]
    factors = (1., 1., 2.)
    c = scipy.sparse.diags_array(weight)
    normal = sum(
        factor * d.T @ c @ d for factor, d in zip(factors, operators)
    )

    def factorize(z_full):
        matrix = normal
        if smoothness:
            matrix = matrix + smoothness * smoothness_operator(mesh, z_full)
        return scipy.sparse.linalg.factorized(scipy.sparse.csc_array(matrix))

    solve = factorize(None)

    z = np.zeros(sides * sides)
    if not pair_rows:
        return z
    rows = np.concatenate(pair_rows)
    weights = np.concatenate(pair_weights)
    curvatures = np.concatenate(pair_curvatures)
    targets = np.concatenate(pair_targets)

    for iteration in range(iterations + 1):
        target_sum = np.zeros((n_interior, 3))
        np.add.at(target_sum, rows, weights[:, None] * targets)
        z_interior = solve(sum(
            factor * d.T @ target_sum[:, k]
            for k, (factor, d) in enumerate(zip(factors, operators))
        ))
        if iteration < iterations:
            hessians = np.stack([d @ z_interior for d in operators], axis=1)
            if slope_correction:
                # Steeper slopes ask for larger determinants, which make
                # the surface steeper still; on large meshes this can
                # run away
                try:
                    with np.errstate(over='raise', invalid='raise'):
                        slopes_squared = sum(
                            (d @ z_interior)**2 for d in gradients
                        )
                        targets = project_to_determinant(
                            hessians[rows],
                            curvatures * (1. + slopes_squared[rows])**2
                        )
                except FloatingPointError as error:
                    raise ValueError(
                        'slope correction diverged; use a smaller mesh scale'
                    ) from error
                if smoothness:
                    z[interior] = z_interior
                    solve = factorize(z)
            else:
                targets = project_to_determinant(hessians[rows], curvatures)

    z[interior] = z_interior
    return z


def edge_residuals(
    kappa_G: npt.NDArray[np.float64],
    points: npt.NDArray[np.float64],
    interior: npt.NDArray[np.bool_],
    network_vertices: npt.NDArray[np.float64],
    network_edges: typing.List[typing.Tuple[int, int]],
    network_curvatures: typing.List[typing.Optional[float]],
    tube_radius: float,
) -> typing.List[typing.Dict[str, typing.Any]]:
    """
    Compare each edge's target curvature to the mean mesh curvature.

    Return one dictionary per edge with the keys `edge`, `target`, and
    `mean_kappa_G` (None when the edge has no target or no interior
    vertices in its tube).
    """
    residuals = []
    for (i, j), curvature in zip(network_edges, network_curvatures):
        members = tube_members(
            points, network_vertices[i, :2], network_vertices[j, :2],
            tube_radius
        )
        members = members[interior[members]]
        residuals.append({
            'edge': [int(i), int(j)],
            'target': None if curvature is None else float(curvature),
            'mean_kappa_G': (
                float(np.mean(kappa_G[members]))
                if curvature is not None and len(members) else None
            ),
        })
    return residuals
