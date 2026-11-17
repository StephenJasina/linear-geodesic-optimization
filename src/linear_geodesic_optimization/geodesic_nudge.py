"""
Greedily nudge the heights of a finished surface so that its geodesics
move toward the network routes they stand for.

Each outer iteration
1. traces the current geodesic of every route the way collation.py
   draws it (a Dijkstra trace through the route's waypoints, tightened
   by `potpourri3d`'s edge-flip solver, which frees it from the
   waypoints; see `route_stretch.RouteStretch.refresh`),
2. measures, at samples along each geodesic, its signed sideways offset
   from the route, plus the offset of each intermediate waypoint from
   the geodesic,
3. linearizes how the geodesic moves sideways under a height change
   `dz` (see `RouteLinearization`),
4. chooses `dz` by minimizing
       Phi_lin(dz) / Phi_0 + lambda_fit * L(z + dz) / L_0
           + mu * mean((dz / relief)^2),
   where Phi_lin is the predicted squared offset, L is the frame's own
   loss (lambda_curvature * L_curvature + lambda_smooth * L_smooth, as
   in the optimizer), and mu is a trust-region weight. `dz` is a
   Gaussian blur of free grid values, so the steps shape broad valleys
   and ridges rather than grid-scale grooves.
5. retraces the geodesics on `z + dz` and keeps the step only if the
   true merit (with the retraced offsets) went down; otherwise mu grows
   and the step is retried.

The sideways motion comes from the discrete length of a geodesic
polyline q_i = p_i + u_i n_i (fixed planar samples p_i and normals n_i,
fixed endpoints), lifted with a local quadratic model of the heights
along each normal: h_i + h_n,i u + h_nn,i u^2 / 2. The geodesic is a
critical point of the length E(u; z), so to first order a height change
moves it by u = -E_uu^{-1} E_uz dz. E_uu is tridiagonal (it is a
discrete Jacobi operator), and h, h_n and h_nn are linear in the
heights (`geodesic_curvature.bilinear_operator` and
`directional_operators`), so E_uz is a sparse matrix.
"""

import typing

import numpy as np
import numpy.typing as npt
import scipy.linalg
import scipy.ndimage
import scipy.optimize
import scipy.sparse

from linear_geodesic_optimization import geodesic_curvature as gc
from linear_geodesic_optimization import pipeline
from linear_geodesic_optimization.mesh.rectangle import Mesh
from linear_geodesic_optimization.route_stretch import RouteStretch, gaussian_blur_matrix


def resample(points: npt.NDArray[np.float64], spacing: float) -> npt.NDArray[np.float64]:
    """
    Resample a planar polyline at equal arc-length steps of about
    `spacing`, keeping both endpoints.
    """
    points = np.asarray(points, dtype=np.float64)[:, :2]
    points = points[np.concatenate((
        [True], np.linalg.norm(np.diff(points, axis=0), axis=1) > 0
    ))]
    arclength = np.concatenate((
        [0.], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))
    ))
    n = max(int(np.ceil(arclength[-1] / spacing)), 2)
    targets = np.linspace(0., arclength[-1], n + 1)
    return np.column_stack([
        np.interp(targets, arclength, points[:, axis]) for axis in range(2)
    ])


def smoothed_normals(points: npt.NDArray[np.float64], width: float) -> npt.NDArray[np.float64]:
    """
    Return unit left normals of a sampled curve, from tangents blurred
    by a Gaussian of standard deviation `width` samples.
    """
    tangents = np.gradient(points, axis=0)
    if width > 0:
        tangents = scipy.ndimage.gaussian_filter1d(tangents, width, axis=0, mode='nearest')
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    return np.column_stack((-tangents[:, 1], tangents[:, 0]))


def closest_points(
    points: npt.NDArray[np.float64],
    polyline: npt.NDArray[np.float64],
) -> typing.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the closest point of `polyline` to each point, and the distances."""
    points = np.asarray(points, dtype=np.float64)[:, :2]
    polyline = np.asarray(polyline, dtype=np.float64)[:, :2]
    best = np.repeat(polyline[:1], len(points), axis=0)
    distances = np.linalg.norm(points - best, axis=1)
    for a, b in zip(polyline[:-1], polyline[1:]):
        direction = b - a
        squared = direction @ direction
        if squared == 0:
            continue
        t = np.clip(((points - a) @ direction) / squared, 0., 1.)
        candidates = a + t[:, None] * direction
        candidate_distances = np.linalg.norm(points - candidates, axis=1)
        closer = candidate_distances < distances
        best[closer] = candidates[closer]
        distances[closer] = candidate_distances[closer]
    return best, distances


class RouteLinearization:
    """
    The first-order sideways motion of one traced geodesic under a
    height change, and the weighted sideways offsets from its route.

    `geodesic` is the traced polyline (from the route's first waypoint
    to its last), `route` the route's waypoints, `weight` the route's
    weight, `spacing` the sample spacing and `normal_smoothing` the
    width (both in the mesh's units) over which the sample normals are
    smoothed. Each intermediate waypoint adds an offset residual
    weighted like `waypoint_weight` times the whole route.

    If a sample lies in a cell touching the mesh boundary, `valid` is
    false and the route is ignored.
    """

    def __init__(
        self,
        mesh: Mesh,
        z: npt.NDArray[np.float64],
        geodesic: npt.NDArray[np.float64],
        route: npt.NDArray[np.float64],
        weight: float,
        spacing: float,
        normal_smoothing: float,
        waypoint_weight: float = 1.,
    ):
        route = np.asarray(route, dtype=np.float64)[:, :2]
        geodesic = np.asarray(geodesic, dtype=np.float64)[:, :2]
        self.route = route
        self.geodesic = geodesic
        q = resample(geodesic, spacing)
        step = np.linalg.norm(q[1] - q[0])
        normals = smoothed_normals(q, normal_smoothing / step)
        self.points = q
        self.normals = normals

        keep, values = gc.bilinear_operator(mesh, q)
        _, (slopes, _, bends) = gc.directional_operators(mesh, q, normals)
        self.valid = bool(keep.all()) and len(q) >= 3
        if not self.valid:
            return

        h, h_n, h_nn = values @ z, slopes @ z, bends @ z
        d = np.diff(q, axis=0)
        e = np.diff(h)
        length = np.sqrt((d**2).sum(axis=1) + e**2)
        normals_a, normals_b = normals[:-1], normals[1:]
        h_n_a, h_n_b = h_n[:-1], h_n[1:]
        g_a = -((d * normals_a).sum(axis=1) + e * h_n_a)
        g_b = (d * normals_b).sum(axis=1) + e * h_n_b

        # Second derivatives of each segment's length in (u_a, u_b)
        m_aa = 1 + h_n_a**2 - e * h_nn[:-1]
        m_bb = 1 + h_n_b**2 + e * h_nn[1:]
        m_ab = -(normals_a * normals_b).sum(axis=1) - h_n_a * h_n_b
        h_aa = m_aa / length - g_a**2 / length**3
        h_bb = m_bb / length - g_b**2 / length**3
        h_ab = m_ab / length - g_a * g_b / length**3
        diagonal = np.zeros(len(q))
        diagonal[:-1] += h_aa
        diagonal[1:] += h_bb
        # Restrict to the interior samples (the endpoints are fixed)
        diagonal = diagonal[1:-1]
        off_diagonal = h_ab[1:-1]
        shift = 0.
        while True:
            banded = np.vstack((np.concatenate(([0.], off_diagonal)), diagonal + shift))
            try:
                self._cholesky = scipy.linalg.cholesky_banded(banded)
                break
            except np.linalg.LinAlgError:
                shift = max(2 * shift, 1e-6 * np.abs(diagonal).max())
        self.hessian_shift = shift

        # Mixed derivatives of the segment slopes dl/du with respect to
        # the per-sample quantities e = h_b - h_a, h_n,a and h_n,b, chained
        # through the operators that give them from the heights
        differences = values[1:] - values[:-1]
        mixed_b = scipy.sparse.diags_array(h_n_b / length - g_b * e / length**3) @ differences \
            + scipy.sparse.diags_array(e / length) @ slopes[1:]
        mixed_a = scipy.sparse.diags_array(-h_n_a / length - g_a * e / length**3) @ differences \
            + scipy.sparse.diags_array(-e / length) @ slopes[:-1]
        # Interior sample i is the end b of segment i - 1 and the end a of
        # segment i
        self._mixed = scipy.sparse.csr_array(mixed_b[:-1] + mixed_a[1:])

        # Residuals: the sideways offset of each interior sample from the
        # route, and of each intermediate waypoint from the geodesic
        interior, interior_normals = q[1:-1], normals[1:-1]
        nearest, _ = closest_points(interior, route)
        sample_length = np.full(len(interior), step)
        indices = [np.arange(len(interior))]
        targets = [((nearest - interior) * interior_normals).sum(axis=1)]
        weights = [weight * sample_length]
        route_length = np.linalg.norm(np.diff(route, axis=0), axis=1).sum()
        for waypoint in route[1:-1]:
            j = int(np.argmin(np.linalg.norm(interior - waypoint, axis=1)))
            indices.append(np.array([j]))
            targets.append(np.array([(waypoint - interior[j]) @ interior_normals[j]]))
            weights.append(np.array([weight * waypoint_weight * route_length]))
        self._indices = np.concatenate(indices)
        self._targets = np.concatenate(targets)
        self._weights = np.concatenate(weights)

    def _solve(self, right):
        return scipy.linalg.cho_solve_banded((self._cholesky, False), right)

    def displacement(self, dz: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return the predicted sideways shift of the interior samples."""
        return -self._solve(self._mixed @ dz)

    def offset(self) -> float:
        """Return the weighted squared offset at the traced geodesic."""
        return float(self._weights @ self._targets**2)

    def evaluate(self, dz) -> typing.Tuple[float, npt.NDArray[np.float64]]:
        """
        Return the predicted weighted squared offset after the height
        change `dz`, and its gradient with respect to `dz`.
        """
        u = self.displacement(dz)
        residual = u[self._indices] - self._targets
        gradient_u = np.zeros_like(u)
        np.add.at(gradient_u, self._indices, 2 * self._weights * residual)
        return (
            float(self._weights @ residual**2),
            -(self._mixed.T @ self._solve(gradient_u)),
        )

    def deviation(self) -> typing.Tuple[float, typing.Optional[float]]:
        """
        Return the largest distance from the geodesic to the route and,
        if the route has intermediate waypoints, the largest distance
        from one of them to the geodesic.
        """
        _, distances = closest_points(self.geodesic, self.route)
        miss = None
        if len(self.route) > 2:
            _, misses = closest_points(self.route[1:-1], self.geodesic)
            miss = float(misses.max())
        return float(distances.max()), miss


def routes_from_output(output, network_vertices):
    """
    Return `(routes, weights)`: the waypoint coordinates of each route of
    an output.json (a route and its reverse counted once) and its
    traffic.
    """
    curves, weights = gc.route_curves(
        output['network'], output['routes'], output['traffic'], 'route'
    )
    return [np.asarray(network_vertices)[list(curve)][:, :2] for curve in curves], weights


class GeodesicNudge:
    """
    The greedy nudge of the module docstring, for one surface.

    `computer` is the frame's loss computer (see `pipeline.computer_from_output`).
    Lengths (`spacing`, `normal_smoothing`, `step_smoothing`) are in the
    mesh's units; they default to half a grid spacing, one grid spacing
    and 0.035 times the mesh scale.
    """

    def __init__(
        self,
        mesh: Mesh,
        routes: typing.Sequence[npt.NDArray[np.float64]],
        weights: npt.NDArray[np.float64],
        computer,
        *,
        lambda_fit: float = 1.,
        step_smoothing: typing.Optional[float] = None,
        spacing: typing.Optional[float] = None,
        normal_smoothing: typing.Optional[float] = None,
        waypoint_weight: float = 1.,
        trust: float = 1.,
        trust_min: float = 1e-3,
        inner_maxiter: int = 100,
        max_rejections: int = 5,
    ):
        self.mesh = mesh
        self.computer = computer
        self.routes = [np.asarray(route, dtype=np.float64)[:, :2] for route in routes]
        weights = np.asarray(weights, dtype=np.float64)
        self.weights = weights / weights.sum()
        sides = mesh.get_width()
        self.sides = sides
        grid_spacing = mesh.get_scale() / (sides - 1)
        self.spacing = 0.5 * grid_spacing if spacing is None else spacing
        self.normal_smoothing = grid_spacing if normal_smoothing is None else normal_smoothing
        step_smoothing = 0.035 * mesh.get_scale() if step_smoothing is None else step_smoothing
        self.blur = gaussian_blur_matrix(sides, step_smoothing / grid_spacing)
        self.lambda_fit = lambda_fit
        self.waypoint_weight = waypoint_weight
        self.trust = trust
        self.trust_min = trust_min
        self.inner_maxiter = inner_maxiter
        self.max_rejections = max_rejections
        self.tracer = RouteStretch(mesh, self.routes, self.weights, self.spacing)

    def trace(self, z) -> typing.List[npt.NDArray[np.float64]]:
        """Return the geodesic of each route on the surface with heights `z`."""
        self.tracer.refresh(np.asarray(z, dtype=np.float64))
        return self.tracer.competitors

    def linearize(self, z) -> typing.List[RouteLinearization]:
        return [
            RouteLinearization(
                self.mesh, z, geodesic, route, weight,
                self.spacing, self.normal_smoothing, self.waypoint_weight,
            )
            for geodesic, route, weight in zip(self.trace(z), self.routes, self.weights)
        ]

    def _blur(self, c):
        return (self.blur @ c.reshape(self.sides, self.sides) @ self.blur.T).reshape(-1)

    def _blur_adjoint(self, gradient):
        return (self.blur.T @ gradient.reshape(self.sides, self.sides) @ self.blur).reshape(-1)

    def _loss(self, z):
        loss = self.computer.forward(z)
        return loss, self.computer.reverse(z)

    def state(self, z, linearizations) -> dict:
        """Return the losses and route deviations at heights `z`."""
        loss, _ = self._loss(z)
        valid = [lin for lin in linearizations if lin.valid]
        deviations = [lin.deviation() for lin in valid]
        scale = self.mesh.get_scale()
        max_deviations = np.array([dev for dev, _ in deviations]) / scale
        misses = np.array([miss for _, miss in deviations if miss is not None]) / scale
        return {
            'offset': float(sum(lin.offset() for lin in valid)),
            'loss': float(loss),
            'L_curvature': float(self.computer.curvature_loss.loss),
            'L_smooth': float(self.computer.smooth_loss.loss),
            'max_deviation_mean': float(max_deviations.mean()),
            'max_deviation_worst': float(max_deviations.max()),
            'miss_mean': float(misses.mean()) if len(misses) else 0.,
            'miss_worst': float(misses.max()) if len(misses) else 0.,
            'invalid_routes': len(linearizations) - len(valid),
        }

    def step(self, z, linearizations, offset_0, loss_0, relief, mu):
        """Return the height change minimizing the linearized merit, and its predicted merit."""
        valid = [lin for lin in linearizations if lin.valid]
        loss_z, _ = self._loss(z)
        n = len(z)

        def objective(c):
            dz = self._blur(c)
            offset, gradient = 0., np.zeros(n)
            for lin in valid:
                value, lin_gradient = lin.evaluate(dz)
                offset += value
                gradient += lin_gradient
            loss, loss_gradient = self._loss(z + dz)
            trust = mu * np.mean((dz / relief)**2)
            value = offset / offset_0 + self.lambda_fit * (loss - loss_z) / loss_0 + trust
            gradient = gradient / offset_0 + self.lambda_fit * loss_gradient / loss_0 \
                + 2 * mu * dz / (n * relief**2)
            return value, self._blur_adjoint(gradient)

        result = scipy.optimize.minimize(
            objective, np.zeros(n), jac=True, method='L-BFGS-B',
            options={'maxiter': self.inner_maxiter},
        )
        dz = self._blur(result.x)
        offset = sum(lin.evaluate(dz)[0] for lin in valid)
        loss, _ = self._loss(z + dz)
        predicted = offset / offset_0 + self.lambda_fit * loss / loss_0
        return dz, predicted, result

    def run(self, z, iterations: int, log=print):
        """
        Nudge the heights `z` for up to `iterations` outer iterations.
        Return the final heights and the history (one entry for the
        start and one per attempted step).
        """
        z = np.asarray(z, dtype=np.float64).copy()
        relief = float(np.ptp(z)) or 1.
        linearizations = self.linearize(z)
        state = self.state(z, linearizations)
        offset_0, loss_0 = state['offset'], state['loss']
        if offset_0 <= 0:
            return z, [dict(state, iteration=0, merit=0.)]
        merit = 1. + self.lambda_fit
        history = [dict(state, iteration=0, merit=merit, accepted=True)]
        log(_format(history[-1]))
        mu = self.trust
        rejections = 0
        for iteration in range(1, iterations + 1):
            dz, predicted, result = self.step(z, linearizations, offset_0, loss_0, relief, mu)
            z_new = z + dz
            linearizations_new = self.linearize(z_new)
            state_new = self.state(z_new, linearizations_new)
            merit_new = state_new['offset'] / offset_0 + self.lambda_fit * state_new['loss'] / loss_0
            accepted = merit_new < merit
            history.append(dict(
                state_new, iteration=iteration, merit=merit_new,
                predicted_merit=float(predicted), accepted=bool(accepted), mu=mu,
                step_rms=float(np.sqrt(np.mean(dz**2))), step_max=float(np.abs(dz).max()),
                inner_iterations=int(result.nit),
            ))
            log(_format(history[-1]))
            if accepted:
                z, linearizations, merit = z_new, linearizations_new, merit_new
                mu = max(mu / 2, self.trust_min)
                rejections = 0
            else:
                mu *= 4
                rejections += 1
                if rejections >= self.max_rejections:
                    break
        return z, history


def _format(entry):
    return (
        f"iteration {entry['iteration']:3d} {'+' if entry.get('accepted') else '-'}"
        f" merit {entry['merit']:.4f}"
        + (f" (predicted {entry['predicted_merit']:.4f}, mu {entry['mu']:.3g},"
           f" step rms {entry['step_rms']:.2e})" if 'predicted_merit' in entry else '')
        + f" | max dev {entry['max_deviation_mean']:.4f} (worst {entry['max_deviation_worst']:.4f})"
        f" miss {entry['miss_mean']:.4f} (worst {entry['miss_worst']:.4f})"
        f" | L_curv {entry['L_curvature']:.4f} L_smooth {entry['L_smooth']:.4g}"
    )
