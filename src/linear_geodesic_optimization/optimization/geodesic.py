"""Module containing utilities to compute geodesic paths."""

import itertools
import time
import typing

import dcelmesh
import numpy as np
import numpy.typing as npt
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg

from linear_geodesic_optimization.mesh.mesh import Mesh
from linear_geodesic_optimization.optimization.laplacian \
    import Computer as Laplacian


class Computer:
    """
    Implementation of the heat method on a triangle mesh.
    """

    def __init__(self, mesh: Mesh, source: int, destinations: typing.List[int],
                 laplacian: Laplacian,
                 m: np.float64 = np.float64(1.)):
        """
        Initialize the computer.

        As input, the computer accepts a mesh and the source vertex
        index to which geodesic distances will be computed.
        Additionally, a third parameter controls the destination
        vertices.

        A fourth parameter is a `laplacian.Computer` corresponding to
        the mesh.

        Finally, an optional parameter `m` controls how smooth the
        computed approximate geodesic distances are.
        """
        self._mesh: Mesh = mesh
        self._topology: dcelmesh.Mesh = mesh.get_topology()
        n = self._topology.n_vertices()
        f = self._topology.n_faces()
        self._laplacian = laplacian

        self._source: int = source
        self._destinations: typing.List[int] = destinations

        # Forward variables
        self._forward_updates: int = self._mesh.get_updates() - 1
        self._coordinates: typing.List[npt.NDArray[np.float64]] = []
        self._m: np.float64 = m
        self._h: np.float64 = np.float64(0.)
        """The mean spacing between mesh nodes."""
        self._t: np.float64 = np.float64(0.)
        """A smoothing parameter."""
        self._delta: npt.NDArray[np.float64] = np.zeros(n)
        """A vector representing the initial heat, indexed by vertex."""
        self._delta[source] = np.float64(1.)
        self._epsilon = 1e-10
        self._LC: sparse.csc_matrix[np.float64] \
            = sparse.csc_matrix((n, n), dtype=np.float64)
        """
        A sparse symmetric matrix representing the cotan Laplacian.
        """
        self._LC_inv = None
        """
        A sparse LU decomposition of LC that can be used in place of
        computing LC's inverse. Note that we subtract a small multiple
        of the identity from LC so that it has nonzero determinant.
        """
        self._S: sparse.csc_matrix[np.float64] \
            = sparse.csc_matrix((n, n), dtype=np.float64)
        """
        A sparse symmetric matrix representing (id - t * Laplacian).
        """
        self._S_inv: typing.Optional[sparse_linalg.SuperLU] = None
        """
        A sparse LU decomposition of S that can be used in place of
        computing S's inverse.
        """
        self._u_neumann: npt.NDArray[np.float64] = np.zeros(n)
        """
        A vector representing the heat at time t, indexed by vertex.
        """
        self._LC_interior: sparse.csc_matrix[np.float64] \
            = sparse.csc_matrix((n, n), dtype=np.float64)
        """
        A sparse symmetric matrix representing the cotan Laplacian.
        """
        self._S_interior: sparse.csc_matrix[np.float64] \
            = sparse.csc_matrix((n, n), dtype=np.float64)
        """
        A sparse symmetric matrix representing (id - t * Laplacian).
        """
        self._S_interior_inv: typing.Optional[sparse_linalg.SuperLU] = None
        """
        A sparse LU decomposition of S that can be used in place of
        computing S's inverse.
        """
        self._u_dirichlet: npt.NDArray[np.float64] = np.zeros(n)
        """
        A vector representing the heat at time t, indexed by vertex.
        """
        self._u: npt.NDArray[np.float64] = np.zeros(n)
        """
        A vector representing the heat at time t, indexed by vertex.
        """
        self._ue: npt.NDArray[np.float64] = np.zeros((f, 3))
        """
        A collection of vectors perpendicular to both the normals of
        each face and the gradient of u, indexed by face.
        """
        self._grad_u: npt.NDArray[np.float64] = np.zeros((f, 3))
        """
        A collection of vectors representing the gradient of u, indexed
        by face.
        """
        self._X: npt.NDArray[np.float64] = np.zeros((f, 3))
        """
        A collection of vectors representing direction of the gradient
        of u, indexed by face.
        """
        self._div_X: npt.NDArray[np.float64] = np.zeros(n)
        """
        A vector representing the divergence of X, indexed by vertex.
        """
        self._phi: npt.NDArray[np.float64] = np.zeros(n)
        """
        A vector representing distances from the source to all mesh
        nodes.
        """
        self.distances: typing.Dict[np.float64] = {
            destination: np.float64(0.)
            for destination in destinations
        }
        """A mapping from destinations to distances from the source."""

        # Reverse variables
        self._reverse_updates: int = self._mesh.get_updates() - 1
        self._partials: npt.NDArray[np.float64] = np.zeros((n, 3))
        self._dif_h: npt.NDArray[np.float64] = np.zeros(n)
        """
        The partials of the mean spacing between mesh nodes, indexed by
        vertex.
        """
        self._dif_t: npt.NDArray[np.float64] = np.zeros(n)
        """
        The partials of the smoothing parameter, indexed by vertex.
        """
        self.dif_distances: typing.Dict[int, np.float64] = {
            destination: np.zeros(n)
            for destination in destinations
        }
        """
        A mapping from destinations to gradients of the distances from
        the source.
        """

    @staticmethod
    def _cross(a, b):
        # For some reason, this is significantly faster than the
        # equivalent numpy function
        return np.array([
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]
        ])

    def forward(self) -> None:
        """
        Compute the forward direction.

        The computed values will be stored in the variable
        `Computer.distances`
        """
        if self._forward_updates == self._mesh.get_updates():
            return
        self._laplacian.forward()
        self._forward_updates = self._mesh.get_updates()
        self._coordinates = list(self._mesh.get_coordinates())
        n = self._topology.n_vertices()
        f = self._topology.n_faces()

        # Compute h and t
        self._h = np.mean([
            np.linalg.norm(
                self._coordinates[halfedge.origin.index]
                - self._coordinates[halfedge.destination.index]
            )
            for halfedge in self._topology.halfedges()
        ])
        self._t = self._m * self._h * self._h

        # Compute LC, S, and u_neumann
        LC_data = []
        LC_row_ind = []
        LC_col_ind = []
        for edge, value in zip(self._topology.edges(),
                               self._laplacian.LC_edges):
            i, j = edge.vertices()
            i, j = i.index, j.index
            LC_data.append(value)
            LC_row_ind.append(i)
            LC_col_ind.append(j)
            LC_data.append(value)
            LC_row_ind.append(j)
            LC_col_ind.append(i)
        for vertex_index, value \
                in enumerate(self._laplacian.LC_vertices):
            LC_data.append(value)
            LC_row_ind.append(vertex_index)
            LC_col_ind.append(vertex_index)
        self._LC = sparse.csc_matrix(
            (LC_data, (LC_row_ind, LC_col_ind)), (n, n)
        )
        self._S = sparse.diags(self._laplacian.D, format='csc') \
            - self._t * self._LC
        self._S_inv = sparse_linalg.splu(self._S)
        self._u_neumann = self._S_inv.solve(self._delta)

        # Compute LC_interior, S_interior, u_dirichlet, and u
        LC_data = []
        LC_row_ind = []
        LC_col_ind = []
        for edge, value in zip(self._topology.edges(),
                               self._laplacian.LC_interior_edges):
            i, j = edge.vertices()
            i, j = i.index, j.index
            LC_data.append(value)
            LC_data.append(value)
            LC_row_ind.append(i)
            LC_row_ind.append(j)
            LC_col_ind.append(j)
            LC_col_ind.append(i)
        for vertex_index, value \
                in enumerate(self._laplacian.LC_interior_vertices):
            LC_data.append(value)
            LC_row_ind.append(vertex_index)
            LC_col_ind.append(vertex_index)
        self._LC_interior = sparse.csc_matrix(
            (LC_data, (LC_row_ind, LC_col_ind)), (n, n)
        )
        self._S_interior = sparse.diags(self._laplacian.D, format='csc') \
            - self._t * self._LC_interior
        self._S_interior_inv = sparse_linalg.splu(self._S_interior)
        self._u_dirichlet = self._S_interior_inv.solve(self._delta)
        self._u = (self._u_neumann + self._u_dirichlet) / 2.

        # Compute ue, grad_u and X
        self._ue = np.zeros((f, 3))
        self._grad_u = np.zeros((f, 3))
        for face_index, (face, N) in enumerate(zip(
            self._topology.faces(), self._laplacian.N
        )):
            for halfedge in face.halfedges():
                origin_index = halfedge.origin.index
                destination_index = halfedge.destination.index
                opposite_index = halfedge.next.destination.index
                self._ue[face_index] += self._u[opposite_index] * (
                    self._coordinates[destination_index]
                    - self._coordinates[origin_index]
                )
            self._grad_u[face_index] = Computer._cross(N, self._ue[face_index])
        norm_grad_u = np.linalg.norm(self._grad_u, axis=1).reshape((-1, 1))
        self._X = np.divide(
            -self._grad_u, norm_grad_u, out=np.zeros_like(self._grad_u),
            where=norm_grad_u != 0
        )

        # Compute div_X
        self._div_X = np.zeros(n)
        for halfedge in self._topology.halfedges():
            origin_index = halfedge.origin.index
            cot_1 = self._laplacian.cot[halfedge.index]
            cot_2 = self._laplacian.cot[halfedge.previous.index]
            e_1 = self._coordinates[halfedge.destination.index] \
                - self._coordinates[origin_index]
            e_2 = self._coordinates[halfedge.next.destination.index] \
                - self._coordinates[origin_index]
            self._div_X[origin_index] += (cot_1 * e_1 + cot_2 * e_2) @ self._X[halfedge.face.index]
        self._div_X /= 2.

        # Compute LC_inv and phi
        self._LC_inv = sparse_linalg.splu(
            self._LC - self._epsilon * sparse.eye(n, format='csc')
        )
        self._phi = self._LC_inv.solve(
            self._div_X - np.mean(self._div_X)
        )
        self._phi -= self._phi[self._source]

        for destination in self._destinations:
            self.distances[destination] = self._phi[destination]

    def _reverse_part(self, vertex: dcelmesh.Mesh.Vertex) \
            -> npt.NDArray[np.float64]:
        """
        Compute the derivative of phi with respect to one of the
        coordinates.
        """
        n = self._topology.n_vertices()
        f = self._topology.n_faces()

        # Compute dif_D
        dif_D_data = np.zeros(n)
        for near in itertools.chain([vertex], vertex.vertices()):
            dif_D_data[near.index] \
                = self._laplacian.dif_D[near.index][vertex.index]
        dif_D = sparse.diags(dif_D_data, format='csc')

        t_start = time.time()
        # print(f'{time.time() - t_start:3.4f}: dif_D computed')

        # Compute dif_LC, dif_S, and dif_u_neumann
        dif_LC_data = []
        dif_LC_row_ind = []
        dif_LC_col_ind = []
        for edge in vertex.edges():
            value = self._laplacian.dif_LC_edges[edge.index][vertex.index]
            i, j = edge.vertices()
            i, j = i.index, j.index
            dif_LC_data.append(value)
            dif_LC_row_ind.append(i)
            dif_LC_col_ind.append(j)
            dif_LC_data.append(value)
            dif_LC_row_ind.append(j)
            dif_LC_col_ind.append(i)
        for halfedge in vertex.halfedges_out():
            halfedge_next = halfedge.next
            edge = halfedge_next.edge
            value = self._laplacian.dif_LC_edges[edge.index][vertex.index]
            i, j = halfedge.destination, halfedge_next.destination
            i, j = i.index, j.index
            dif_LC_data.append(value)
            dif_LC_row_ind.append(i)
            dif_LC_col_ind.append(j)
            dif_LC_data.append(value)
            dif_LC_row_ind.append(j)
            dif_LC_col_ind.append(i)
        for near in itertools.chain([vertex], vertex.vertices()):
            dif_LC_data.append(
                self._laplacian.dif_LC_vertices[near.index][vertex.index]
            )
            dif_LC_row_ind.append(near.index)
            dif_LC_col_ind.append(near.index)
        dif_LC = sparse.csc_matrix(
            (dif_LC_data, (dif_LC_row_ind, dif_LC_col_ind)), (n, n)
        )
        dif_S = dif_D - (self._dif_t[vertex.index] * self._LC
                         + self._t * dif_LC)
        dif_u_neumann = -self._S_inv.solve(dif_S @ self._u_neumann)

        # Compute dif_LC_interior, dif_S_interior, and dif_u_dirichlet
        dif_LC_data = []
        dif_LC_row_ind = []
        dif_LC_col_ind = []
        for edge in vertex.edges():
            i, j = edge.vertices()
            value = self._laplacian.dif_LC_interior_edges[edge.index][vertex.index]
            i, j = i.index, j.index
            dif_LC_data.append(value)
            dif_LC_row_ind.append(i)
            dif_LC_col_ind.append(j)
            dif_LC_data.append(value)
            dif_LC_row_ind.append(j)
            dif_LC_col_ind.append(i)
        for halfedge in vertex.halfedges_out():
            halfedge_next = halfedge.next
            i, j = halfedge.destination, halfedge_next.destination
            edge = halfedge_next.edge
            value = self._laplacian.dif_LC_interior_edges[edge.index][vertex.index]
            i, j = i.index, j.index
            dif_LC_data.append(value)
            dif_LC_row_ind.append(i)
            dif_LC_col_ind.append(j)
            dif_LC_data.append(value)
            dif_LC_row_ind.append(j)
            dif_LC_col_ind.append(i)
        for near in itertools.chain([vertex], vertex.vertices()):
            dif_LC_data.append(
                self._laplacian.dif_LC_interior_vertices[near.index][vertex.index]
            )
            dif_LC_row_ind.append(near.index)
            dif_LC_col_ind.append(near.index)
        dif_LC_interior = sparse.csc_matrix(
            (dif_LC_data, (dif_LC_row_ind, dif_LC_col_ind)), (n, n)
        )
        dif_S_interior = dif_D - (self._dif_t[vertex.index] * self._LC_interior
                         + self._t * dif_LC_interior)
        dif_u_dirichlet = -self._S_interior_inv.solve(dif_S_interior @ self._u_dirichlet)

        # Compute dif_u
        dif_u = (dif_u_neumann + dif_u_dirichlet) / 2.

        # print(f'{time.time() - t_start:3.4f}: dif_u computed')

        # Compute dif_ue
        dif_ue = []
        for face in self._topology.faces():
            i, j, k = face.vertices()
            i, j, k = i.index, j.index, k.index
            dif_ue.append(
                dif_u[i] * (self._coordinates[k] - self._coordinates[j])
                + dif_u[j] * (self._coordinates[i] - self._coordinates[k])
                + dif_u[k] * (self._coordinates[j] - self._coordinates[i])
            )
        for halfedge in vertex.halfedges_out():
            dif_ue[halfedge.face.index] \
                -= self._u[halfedge.next.destination.index] \
                    * self._partials[vertex.index]
        for halfedge in vertex.halfedges_in():
            dif_ue[halfedge.face.index] \
                += self._u[halfedge.next.destination.index] \
                    * self._partials[vertex.index]
        dif_ue = np.array(dif_ue)

        # print(f'{time.time() - t_start:3.4f}: dif_ue computed')

        # Compute dif_grad_u
        dif_grad_u = np.cross(self._laplacian.N, dif_ue)
        for face in vertex.faces():
            face_index = face.index
            dif_grad_u[face_index] += Computer._cross(
                self._laplacian.dif_N[face_index][vertex.index],
                self._ue[face_index]
            )

        # Compute dif_X
        dif_X = (np.sum(self._X * dif_grad_u, axis=1).reshape((-1, 1))
                 * self._X
                 - dif_grad_u) \
            / np.linalg.norm(self._grad_u, axis=1).reshape((-1, 1))

        # print(f'{time.time() - t_start:3.4f}: dif_X computed')

        # Compute dif_div_X
        dif_div_X = np.zeros(n)
        for halfedge in self._topology.halfedges():
            origin_index = halfedge.origin.index
            cot_1 = self._laplacian.cot[halfedge.index]
            cot_2 = self._laplacian.cot[halfedge.previous.index]
            e_1 = self._coordinates[halfedge.destination.index] \
                - self._coordinates[origin_index]
            e_2 = self._coordinates[halfedge.next.destination.index] \
                - self._coordinates[origin_index]
            dif_div_X[origin_index] += (cot_1 * e_1 + cot_2 * e_2) @ dif_X[halfedge.face.index]
        for face in vertex.faces():
            for halfedge in face.halfedges():
                origin_index = halfedge.origin.index
                dif_cot_1 = self._laplacian.dif_cot[halfedge.index][vertex.index]
                dif_cot_2 = self._laplacian.dif_cot[halfedge.previous.index][vertex.index]
                e_1 = self._coordinates[halfedge.destination.index] \
                    - self._coordinates[origin_index]
                e_2 = self._coordinates[halfedge.next.destination.index] \
                    - self._coordinates[origin_index]
                dif_div_X[origin_index] \
                    += (dif_cot_1 * e_1 + dif_cot_2 * e_2) @ self._X[face.index]
        for halfedge in vertex.halfedges_out():
            cot_1 = self._laplacian.cot[halfedge.index]
            cot_2 = self._laplacian.cot[halfedge.previous.index]
            multiplier = self._partials[vertex.index] @ self._X[halfedge.face.index]
            dif_div_X[vertex.index] -= (cot_1 + cot_2) * multiplier
            dif_div_X[halfedge.destination.index] += cot_1 * multiplier
            dif_div_X[halfedge.next.destination.index] += cot_2 * multiplier
        dif_div_X /= 2.

        # print(f'{time.time() - t_start:3.4f}: dif_div_X computed')

        dif_phi = self._LC_inv.solve(dif_div_X - np.mean(dif_div_X) - dif_LC @ self._phi)
        dif_phi -= dif_phi[self._source]

        # print(f'{time.time() - t_start:3.4f}: dif_phi computed')

        return dif_phi

    def reverse(self) -> None:
        """
        Compute the reverse direction (that is, partials).

        The computed values will be stored in the variable
        `Computer.dif_distances`.
        """
        self.forward()
        if self._reverse_updates == self._mesh.get_updates():
            return
        self._laplacian.reverse()
        self._reverse_updates = self._mesh.get_updates()
        self._partials = self._mesh.get_partials()

        n = self._topology.n_vertices()
        he = self._topology.n_halfedges()
        f = self._topology.n_faces()

        # Compute dif_h and dif_t
        self._dif_h = np.zeros(n)
        for halfedge in self._topology.halfedges():
            origin_index = halfedge.origin.index
            destination_index = halfedge.destination.index
            origin_coordinate = self._coordinates[origin_index]
            destination_coordinate = self._coordinates[destination_index]
            e = destination_coordinate - origin_coordinate
            e /= np.linalg.norm(e)
            self._dif_h[origin_index] -= e @ self._partials[origin_index]
            self._dif_h[destination_index] += e @ self._partials[destination_index]
        self._dif_h /= he
        self._dif_t = 2 * self._m * self._h * self._dif_h

        self.dif_distances = {
            destination: np.zeros(n)
            for destination in self._destinations
        }
        for vertex_index, vertex in enumerate(self._topology.vertices()):
            dif_phi = self._reverse_part(vertex)
            for destination in self._destinations:
                self.dif_distances[destination][vertex_index] \
                    = dif_phi[destination]
