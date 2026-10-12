"""
Module containing utilities to compute the tensor smoothness loss.

The loss is a discrete Dirichlet energy of the shape operator:
    A_total * sum_{interior edges uv} w_uv ||S_u - R_uv S_v R_uv^T||_F^2,
where w_uv are the cotangent weights, S_v is the shape operator at v
written as a 3x3 matrix in ambient coordinates, and R_uv is the minimal
rotation taking the normal at v to the normal at u. Unlike the MVS-cross
energy, this never takes eigenvalues, so it is smooth everywhere
(including at umbilic points such as a flat mesh). For small slopes, it
reduces to the integral of ||grad Hess f||^2.

The shape operators are estimated with finite differences, so this
requires an untrimmed `RectangleMesh`. Only the estimator is specific to
the grid; the energy itself applies to any mesh.

See `doc/computation/smooth_loss.tex` for the derivation.
"""

import typing

import numpy as np
import numpy.typing as npt
import scipy.sparse

from linear_geodesic_optimization.construction import hessian_design
from linear_geodesic_optimization.mesh.mesh import Mesh
from linear_geodesic_optimization.optimization.laplacian \
    import Computer as Laplacian


def grid_operators(mesh: Mesh) -> typing.Tuple[
    npt.NDArray[np.int64], typing.List[scipy.sparse.csr_array]
]:
    """
    Return the interior vertex indices and the finite-difference
    operators `[Dx, Dy, Dxx, Dyy, Dxy]` of a rectangle mesh.

    Each operator maps heights on all vertices to values on the interior
    vertices.
    """
    sides = mesh.get_width()
    if mesh.get_height() != sides \
            or mesh.get_topology().n_vertices != sides * sides:
        raise ValueError(
            'The tensor smoothness loss needs an untrimmed square '
            'rectangle mesh'
        )
    scale = mesh.get_scale()
    dxx, dyy, dxy, interior = hessian_design.hessian_operators(sides, scale)
    dx, dy = hessian_design.gradient_operators(sides, scale)
    return np.flatnonzero(interior), [dx, dy, dxx, dyy, dxy]


def frames(
    gradients: npt.NDArray[np.float64]
) -> typing.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64],
                  npt.NDArray[np.float64]]:
    """
    Return `(P, W, N)` for a height field with the given gradients.

    Here, `W = sqrt(1 + |grad f|^2)`, `N = (-grad f, 1) / W` is the
    unit normal, and `P = J g^{-1}` is the 3x2 map from the parameter
    domain to ambient coordinates that turns the second fundamental form
    into the ambient shape operator. Explicitly,
    `P = [I - p p^T / W^2; p^T / W^2]` with `p = grad f`.
    """
    s = 1. / (1. + np.sum(gradients**2, axis=1))
    w = 1. / np.sqrt(s)
    p = np.zeros((len(gradients), 3, 2))
    p[:, :2, :] = np.eye(2) \
        - s[:, None, None] * gradients[:, :, None] * gradients[:, None, :]
    p[:, 2, :] = s[:, None] * gradients
    normals = np.concatenate(
        (-gradients, np.ones((len(gradients), 1))), axis=1
    ) / w[:, None]
    return p, w, normals


def shape_operators(
    p: npt.NDArray[np.float64],
    w: npt.NDArray[np.float64],
    hessians: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Return the ambient shape operators `P (Hess f / W) P^T`.

    The Hessians are given as rows `(f_xx, f_yy, f_xy)`.
    """
    m = np.empty((len(hessians), 2, 2))
    m[:, 0, 0] = hessians[:, 0]
    m[:, 1, 1] = hessians[:, 1]
    m[:, 0, 1] = m[:, 1, 0] = hessians[:, 2]
    return p @ (m / w[:, None, None]) @ np.swapaxes(p, 1, 2)


def transports(
    normals_u: npt.NDArray[np.float64],
    normals_v: npt.NDArray[np.float64],
) -> typing.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64],
                  npt.NDArray[np.float64]]:
    """
    Return `(R, c, d)`, where `R` is the minimal rotation taking each
    normal at v to the corresponding normal at u.

    With `c = n_v x n_u` and `d = n_v . n_u`,
    `R = I + [c]_x + (c c^T - |c|^2 I) / (1 + d)`.
    """
    c = np.cross(normals_v, normals_u)
    d = np.sum(normals_v * normals_u, axis=1)
    cross = np.zeros((len(c), 3, 3))
    cross[:, 0, 1], cross[:, 0, 2] = -c[:, 2], c[:, 1]
    cross[:, 1, 0], cross[:, 1, 2] = c[:, 2], -c[:, 0]
    cross[:, 2, 0], cross[:, 2, 1] = -c[:, 1], c[:, 0]
    square = c[:, :, None] * c[:, None, :] \
        - np.sum(c**2, axis=1)[:, None, None] * np.eye(3)
    return np.eye(3) + cross + square / (1. + d)[:, None, None], c, d


class Computer:
    """Implementation of the tensor smoothness loss."""

    def __init__(self, mesh: Mesh, laplacian: Laplacian):
        """Initialize the computer."""
        self._mesh = mesh
        self._topology = mesh.get_topology()
        self._laplacian: Laplacian = laplacian

        self._interior, operators = grid_operators(mesh)
        self._dx, self._dy, self._dxx, self._dyy, self._dxy = operators
        self._row_of = np.full(self._topology.n_vertices, -1)
        self._row_of[self._interior] = np.arange(len(self._interior))

        edges = np.array([
            [vertex.index for vertex in edge.vertices()]
            for edge in self._topology.edges()
        ])
        self._edges = np.flatnonzero(
            (self._row_of[edges[:, 0]] >= 0) & (self._row_of[edges[:, 1]] >= 0)
        )
        self._u = self._row_of[edges[self._edges, 0]]
        self._v = self._row_of[edges[self._edges, 1]]

        # Forward variables
        self._forward_updates: int = mesh.get_updates() - 1
        self.loss: np.float64 = np.float64(0.)
        """The tensor smoothness loss."""

        # Reverse variables
        self._reverse_updates: int = mesh.get_updates() - 1
        self.dif_loss: npt.NDArray[np.float64] \
            = np.zeros(self._topology.n_vertices)
        """The partials of the smoothness loss, indexed by vertices."""

    def forward(self) -> None:
        """
        Compute the forward direction.

        The computed values will be stored in the variables:
        * `Computer.loss`
        """
        if self._forward_updates == self._mesh.get_updates():
            return
        self._laplacian.forward()
        self._forward_updates = self._mesh.get_updates()

        z = self._mesh.get_parameters()
        self._gradients = np.stack([self._dx @ z, self._dy @ z], axis=1)
        self._hessians = np.stack(
            [self._dxx @ z, self._dyy @ z, self._dxy @ z], axis=1
        )
        self._p, self._w, self._normals = frames(self._gradients)
        self._s = shape_operators(self._p, self._w, self._hessians)
        self._r, self._c, self._d = transports(
            self._normals[self._u], self._normals[self._v]
        )
        self._difference = self._s[self._u] \
            - self._r @ self._s[self._v] @ np.swapaxes(self._r, 1, 2)
        self._squared_norms = np.sum(self._difference**2, axis=(1, 2))
        self._weights = np.asarray(self._laplacian.LC_interior_edges)[self._edges]
        self._energy = self._weights @ self._squared_norms
        self._total_area = np.sum(self._laplacian.A)

        self.loss = self._total_area * self._energy

    def reverse(self) -> None:
        """
        Compute the reverse direction (that is, partials).

        The computed values will be stored in the variables:
        * `Computer.dif_loss`
        """
        if self._reverse_updates == self._mesh.get_updates():
            return
        self.forward()
        self._laplacian.reverse()
        self._reverse_updates = self._mesh.get_updates()

        n_interior = len(self._interior)
        r, r_t = self._r, np.swapaxes(self._r, 1, 2)
        s_v = self._s[self._v]

        # Partials with respect to the per-edge differences, shape
        # operators, and transports
        dif_difference = 2. * self._total_area \
            * self._weights[:, None, None] * self._difference
        dif_s = np.zeros((n_interior, 3, 3))
        np.add.at(dif_s, self._u, dif_difference)
        np.add.at(dif_s, self._v, -r_t @ dif_difference @ r)
        dif_r = -2. * dif_difference @ r @ s_v

        # Through R = I + [c]_x + (c c^T - |c|^2 I) / (1 + d)
        c, d = self._c, self._d
        q = 1. / (1. + d)
        dif_c = np.stack([
            dif_r[:, 2, 1] - dif_r[:, 1, 2],
            dif_r[:, 0, 2] - dif_r[:, 2, 0],
            dif_r[:, 1, 0] - dif_r[:, 0, 1],
        ], axis=1)
        dif_c += q[:, None] * (
            np.einsum('eij,ej->ei', dif_r + np.swapaxes(dif_r, 1, 2), c)
            - 2. * np.trace(dif_r, axis1=1, axis2=2)[:, None] * c
        )
        square = c[:, :, None] * c[:, None, :] \
            - np.sum(c**2, axis=1)[:, None, None] * np.eye(3)
        dif_d = -q**2 * np.sum(dif_r * square, axis=(1, 2))

        # Through c = n_v x n_u and d = n_v . n_u
        normals_u, normals_v = self._normals[self._u], self._normals[self._v]
        dif_normals = np.zeros((n_interior, 3))
        np.add.at(
            dif_normals, self._u,
            np.cross(dif_c, normals_v) + dif_d[:, None] * normals_v
        )
        np.add.at(
            dif_normals, self._v,
            np.cross(normals_u, dif_c) + dif_d[:, None] * normals_u
        )

        # Through S = P M P^T with M = Hess f / W
        p, w, gradients = self._p, self._w, self._gradients
        hessians = np.empty((n_interior, 2, 2))
        hessians[:, 0, 0] = self._hessians[:, 0]
        hessians[:, 1, 1] = self._hessians[:, 1]
        hessians[:, 0, 1] = hessians[:, 1, 0] = self._hessians[:, 2]
        m = hessians / w[:, None, None]
        dif_m = np.swapaxes(p, 1, 2) @ dif_s @ p
        dif_p = (dif_s + np.swapaxes(dif_s, 1, 2)) @ p @ m
        dif_hessians = dif_m / w[:, None, None]
        dif_inverse_w = np.sum(dif_m * hessians, axis=(1, 2))

        # Through P, N, and 1 / W as functions of p = grad f, using
        # s = 1 / W^2 and d(1 / W) / dp = -p / W^3
        s = 1. / w**2
        dif_top = dif_p[:, :2, :]
        dif_bottom = dif_p[:, 2, :]
        dif_gradients = -s[:, None] * np.einsum(
            'eij,ej->ei', dif_top + np.swapaxes(dif_top, 1, 2), gradients
        )
        dif_gradients += 2. * (s**2 * np.einsum(
            'ei,eij,ej->e', gradients, dif_top, gradients
        ))[:, None] * gradients
        dif_gradients += s[:, None] * dif_bottom \
            - 2. * (s**2 * np.sum(dif_bottom * gradients, axis=1))[:, None] \
            * gradients
        dif_normals_xy = dif_normals[:, :2]
        dif_gradients += -dif_normals_xy / w[:, None] \
            + gradients * (
                np.sum(dif_normals_xy * gradients, axis=1) / w**3
            )[:, None]
        dif_gradients -= ((dif_normals[:, 2] + dif_inverse_w) / w**3)[:, None] \
            * gradients

        self.dif_loss = self._dx.T @ dif_gradients[:, 0] \
            + self._dy.T @ dif_gradients[:, 1] \
            + self._dxx.T @ dif_hessians[:, 0, 0] \
            + self._dyy.T @ dif_hessians[:, 1, 1] \
            + self._dxy.T @ (dif_hessians[:, 0, 1] + dif_hessians[:, 1, 0])

        # Through the cotangent weights and the total area
        dif_lc = self._laplacian.dif_LC_interior_edges
        for edge_index, squared_norm in zip(self._edges, self._squared_norms):
            for index, dif_weight in dif_lc[edge_index].items():
                self.dif_loss[index] += self._total_area * squared_norm \
                    * dif_weight
        for dif_A_part in self._laplacian.dif_A:
            for index, dif_A_element in dif_A_part.items():
                self.dif_loss[index] += self._energy * dif_A_element
