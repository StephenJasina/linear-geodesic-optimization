"""Module containing utilities to compute the Laplace-Beltrami operator."""

import itertools
import typing

import dcelmesh
import numpy as np
import numpy.typing as npt
from scipy import sparse

from linear_geodesic_optimization.mesh.mesh import Mesh


class Computer:
    """Implementation of the Laplace-Beltrami operator on a mesh."""

    def __init__(self, mesh: Mesh):
        """Initialize the computer."""
        self._mesh: Mesh = mesh
        self._topology: dcelmesh.Mesh = mesh.get_topology()

        n_v = self._topology.n_vertices
        n_he = self._topology.n_halfedges
        n_e = self._topology.n_edges
        n_f = self._topology.n_faces

        # Mappings between different types of mesh elements. These are
        # used for vectorized operations. Note: the naming convention is
        # tecnically backwards.
        self._v_i_to_he = np.array([halfedge.origin.index for halfedge in self._topology.halfedges()])
        self._v_j_to_he = np.array([halfedge.destination.index for halfedge in self._topology.halfedges()])
        self._v_k_to_he = np.array([halfedge.next.destination.index for halfedge in self._topology.halfedges()])
        self._v_i_to_f, self._v_j_to_f, self._v_k_to_f = np.array([
            *zip(*[
                [halfedge.origin.index for halfedge in face.halfedges()]
                for face in self._topology.faces()
            ])
        ])
        self._he_to_v_i = sparse.csr_array((
            [1] * n_he,
            (
                [halfedge.origin.index for halfedge in self._topology.halfedges()],
                [halfedge.index for halfedge in self._topology.halfedges()]
            )
        ), (n_v, n_he))
        self._he_to_v_j = sparse.csr_array((
            [1] * n_he,
            (
                [halfedge.destination.index for halfedge in self._topology.halfedges()],
                [halfedge.index for halfedge in self._topology.halfedges()]
            )
        ), (n_v, n_he))
        halfedges_interior = list(filter(
            lambda halfedge: not halfedge.origin.is_on_boundary() and not halfedge.destination.is_on_boundary(),
            self._topology.halfedges()
        ))
        self._he_to_v_i_interior = sparse.csr_array((
            [1 for halfedge in halfedges_interior],
            (
                [halfedge.origin.index for halfedge in halfedges_interior],
                [halfedge.index for halfedge in halfedges_interior]
            )
        ), (n_v, n_he))
        self._he_to_v_j_interior = sparse.csr_array((
            [1 for halfedge in halfedges_interior],
            (
                [halfedge.destination.index for halfedge in halfedges_interior],
                [halfedge.index for halfedge in halfedges_interior]
            )
        ), (n_v, n_he))
        self._he_to_e = sparse.csr_array((
            [1] * n_he,
            (
                [halfedge.edge.index for halfedge in self._topology.halfedges()],
                [halfedge.index for halfedge in self._topology.halfedges()]
            )
        ), (n_e, n_he))
        self._he_to_e_interior = sparse.csr_array((
            [1 for halfedge in halfedges_interior],
            (
                [halfedge.edge.index for halfedge in halfedges_interior],
                [halfedge.index for halfedge in halfedges_interior]
            )
        ), (n_e, n_he))
        self._he_to_f = sparse.csr_array((
            [1] * n_he,
            (
                [halfedge.face.index for halfedge in self._topology.halfedges()],
                [halfedge.index for halfedge in self._topology.halfedges()]
            )
        ), (n_f, n_he))
        self._f_to_v = sparse.csr_array((
            [1] * n_he,
            (
                [halfedge.origin.index for halfedge in self._topology.halfedges()],
                [halfedge.face.index for halfedge in self._topology.halfedges()]
            )
        ), (n_v, n_f))
        self._f_to_he = sparse.csr_array((
            [1] * n_he,
            (
                [halfedge.index for halfedge in self._topology.halfedges()],
                [halfedge.face.index for halfedge in self._topology.halfedges()]
            )
        ), (n_he, n_f))
        self._f_to_he_i, self._f_to_he_j, self._f_to_he_k = np.array([
            *zip(*[
                [halfedge.index for halfedge in face.halfedges()]
                for face in self._topology.faces()
            ])
        ])

        # Forward variables
        self._forward_updates: int = mesh.get_updates() - 1
        self._coordinates: npt.NDArray[np.float64] = np.zeros((n_v, 3))

        self.N: npt.NDArray[np.float64] = np.zeros((n_f, 3))
        """An array of normals of faces, indexed by faces."""
        self.A: npt.NDArray[np.float64] = np.zeros(n_f)
        """An array areas of faces, indexed by faces."""
        self.D: npt.NDArray[np.float64] = np.zeros(n_v)
        """An array of vertex areas, indexed by vertices."""
        self.cot: npt.NDArray[np.float64] = np.zeros(n_he)
        """
        An array of cotangents of the opposing angles to halfedges,
        indexed by halfedges.
        """
        self.LC_edges: npt.NDArray[np.float64] = np.zeros(n_e)
        """
        An array of (non-trivial) off-diagonal entries of the
        Laplace-Beltrami operator, indexed by edges.
        """
        self.LC_vertices: typing.List[np.float64] = np.zeros(n_v)
        """
        An array of diagonal entries of the Laplace-Beltrami operator,
        indexed by vertices.
        """
        self.LC_interior_edges: npt.NDArray[np.float64] = np.zeros(n_e)
        """
        An array of (non-trivial) off-diagonal entries of the
        Laplace-Beltrami operator ignoring the boundary vertices,
        indexed by edges.
        """
        self.LC_interior_vertices: npt.NDArray[np.float64] = np.zeros(n_v)
        """
        An array of diagonal entries of the Laplace-Beltrami operator
        ignoring the boundary vertices, indexed by vertices.
        """

        # Reverse variables
        self._reverse_updates: int = mesh.get_updates() - 1
        self._partials: npt.NDArray[np.float64] \
            = np.zeros((self._topology.n_vertices, 3))
        self.dif_N: typing.List[typing.Dict[int, npt.NDArray[np.float64]]] \
            = [{} for _ in range(self._topology.n_faces)]
        """
        A list of partials of normals of faces, indexed by faces and
        then by (incident) vertices.
        """
        self.dif_A: typing.List[typing.Dict[int, np.float64]] \
            = [{} for _ in range(self._topology.n_faces)]
        """
        A list partials of areas of faces, indexed by faces and then by
        (incident) vertices.
        """
        self.dif_D: typing.List[typing.Dict[int, np.float64]] \
            = [{} for _ in range(self._topology.n_vertices)]
        """
        A list of partials of vertex areas, indexed by vertices and
        then by vertices (at most distance 1 away).
        """
        self.dif_cot: typing.List[typing.Dict[int, np.float64]] \
            = [{} for _ in range(self._topology.n_halfedges)]
        """
        A list of partials of cotangents of the opposing angles to
        halfedges, indexed by halfedges and then by vertices (of the
        same face).
        """
        self.dif_LC_edges: \
            typing.List[typing.Dict[int, np.float64]] \
            = [{} for _ in range(self._topology.n_edges)]
        """
        A list of partials of (non-trivial) off-diagonal entries of the
        Laplace-Beltrami operator, indexed by edges and then by vertices
        (of the same and opposing faces).
        """
        self.dif_LC_vertices: \
            typing.List[typing.Dict[int, np.float64]] \
            = [{} for _ in range(self._topology.n_vertices)]
        """
        A list of partials of diagonal entries of the Laplace-Beltrami
        operator, indexed by vertices and then vertices (at most
        distance 1 away).
        """
        self.dif_LC_interior_edges: \
            typing.List[typing.Dict[int, np.float64]] \
            = [{} for _ in range(self._topology.n_edges)]
        """
        A list of partials of (non-trivial) off-diagonal entries of the
        Laplace-Beltrami operator ignoring the boundary vertices,
        indexed by edges and then by vertices (of the same and opposing
        faces).
        """
        self.dif_LC_interior_vertices: \
            typing.List[typing.Dict[int, np.float64]] \
            = [{} for _ in range(self._topology.n_vertices)]
        """
        A list of partials of diagonal entries of the Laplace-Beltrami
        operator ignoring the boundary vertices, indexed by vertices and
        then vertices (at most distance 1 away).
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

        The computed values will be stored in the variables:
        * `Computer.N`
        * `Computer.A`
        * `Computer.D`
        * `Computer.cot`
        * `Computer.LC_halfedges`
        * `Computer.LC_vertices`
        * `Computer.LC_interior_halfedges`
        * `Computer.LC_interior_vertices`
        """
        if self._forward_updates == self._mesh.get_updates():
            return
        self._forward_updates = self._mesh.get_updates()
        self._coordinates = self._mesh.get_coordinates()

        self.N = np.cross(self._coordinates[self._v_i_to_f, :] - self._coordinates[self._v_k_to_f, :], self._coordinates[self._v_j_to_f, :] - self._coordinates[self._v_k_to_f, :], axis=1)
        self.A = np.linalg.norm(self.N, axis=1) / 2.
        self.D = self._f_to_v @ self.A / 3.
        self.cot = np.sum((self._coordinates[self._v_i_to_he, :] - self._coordinates[self._v_k_to_he, :]) * (self._coordinates[self._v_j_to_he, :] - self._coordinates[self._v_k_to_he, :]), axis=1) / (2. * self._f_to_he @ self.A)
        self.LC_edges = self._he_to_e @ self.cot / 2.
        self.LC_vertices = -(self._he_to_v_i @ self.cot + self._he_to_v_j @ self.cot) / 2.
        self.LC_interior_edges = self._he_to_e_interior @ self.cot / 2.
        self.LC_interior_vertices = -(self._he_to_v_i_interior @ self.cot + self._he_to_v_j_interior @ self.cot) / 2.

    def reverse(self) -> None:
        """
        Compute the reverse direction (that is, partials).

        The computed values will be stored in the variables:
        * `Computer.dif_N`
        * `Computer.dif_A`
        * `Computer.dif_D`
        * `Computer.dif_cot`
        * `Computer.dif_LC_halfedges`
        * `Computer.dif_LC_vertices`
        * `Computer.dif_LC_interior_halfedges`
        * `Computer.dif_LC_interior_vertices`
        """
        if self._reverse_updates == self._mesh.get_updates():
            return
        self.forward()
        self._reverse_updates = self._mesh.get_updates()
        self._partials = self._mesh.get_partials()

        # Reset quantities that will be computed via accumulation
        self.dif_D = [
            {
                near.index: np.float64(0.)
                for near in itertools.chain([vertex], vertex.vertices())
            }
            for vertex in self._topology.vertices()
        ]
        self.dif_LC_edges = [
            {
                vertex.index: np.float64(0.)
                for face in edge.faces()
                for vertex in face.vertices()
            }
            for edge in self._topology.edges()
        ]
        self.dif_LC_vertices = [
            {
                near.index: np.float64(0.)
                for near in itertools.chain([vertex], vertex.vertices())
            }
            for vertex in self._topology.vertices()
        ]
        self.dif_LC_interior_edges = [
            {
                vertex.index: np.float64(0.)
                for face in edge.faces()
                for vertex in face.vertices()
            }
            for edge in self._topology.edges()
        ]
        self.dif_LC_interior_vertices = [
            {
                near.index: np.float64(0.)
                for near in itertools.chain([vertex], vertex.vertices())
            }
            for vertex in self._topology.vertices()
        ]

        # Use 3 separate arrays here because SciPy's `sparse` library
        # doesn't currently support arrays of dimension greater than 2.
        dif_N_i = np.cross(self._coordinates[self._v_k_to_f] - self._coordinates[self._v_j_to_f], self._partials[self._v_i_to_f], axis=1)
        dif_N_j = np.cross(self._coordinates[self._v_i_to_f] - self._coordinates[self._v_k_to_f], self._partials[self._v_j_to_f], axis=1)
        dif_N_k = np.cross(self._coordinates[self._v_j_to_f] - self._coordinates[self._v_i_to_f], self._partials[self._v_k_to_f], axis=1)

        # TODO: Remove this
        for face, dif_N_i_part, dif_N_j_part, dif_N_k_part in zip(self._topology.faces(), dif_N_i, dif_N_j, dif_N_k):
            halfedges = list(face.halfedges())
            self.dif_N[face.index][halfedges[0].origin.index] = dif_N_i_part
            self.dif_N[face.index][halfedges[1].origin.index] = dif_N_j_part
            self.dif_N[face.index][halfedges[2].origin.index] = dif_N_k_part

        self.dif_A = np.zeros(self._topology.n_halfedges)
        self.dif_A[self._f_to_he_i] = np.sum(self.N * dif_N_i, axis=1) / (4. * self.A)
        self.dif_A[self._f_to_he_j] = np.sum(self.N * dif_N_j, axis=1) / (4. * self.A)
        self.dif_A[self._f_to_he_k] = np.sum(self.N * dif_N_k, axis=1) / (4. * self.A)

        # self.dif_D = self._he_to_f @ self.dif_A / 3.

        for halfedge in self._topology.halfedges():
            u = halfedge.origin
            v = halfedge.destination
            w = halfedge.previous.origin

            # Set dif_D
            third_dif_A_u = self.dif_A[halfedge.index] / 3.
            self.dif_D[u.index][u.index] += third_dif_A_u
            self.dif_D[v.index][u.index] += third_dif_A_u
            self.dif_D[w.index][u.index] += third_dif_A_u

        # Need a separate loop here to ensure dif_A has been computed
        for halfedge in self._topology.halfedges():
            u = halfedge.origin
            v = halfedge.destination
            w = halfedge.previous.origin
            pu = self._coordinates[u.index]
            pv = self._coordinates[v.index]
            pw = self._coordinates[w.index]

            # Set dif_cot
            area = self.A[halfedge.face.index]
            cotangent = self.cot[halfedge.index]
            dif_cot_u = ((pv - pw) @ self._partials[u.index]
                         - 2. * cotangent * self.dif_A[halfedge.index]) / (2. * area)
            dif_cot_v = ((pu - pw) @ self._partials[v.index]
                         - 2. * cotangent * self.dif_A[halfedge.next.index]) / (2. * area)
            dif_cot_w = ((2. * pw - pu - pv) @ self._partials[w.index]
                         - 2. * cotangent * self.dif_A[halfedge.previous.index]) / (2. * area)
            self.dif_cot[halfedge.index][u.index] = dif_cot_u
            self.dif_cot[halfedge.index][v.index] = dif_cot_v
            self.dif_cot[halfedge.index][w.index] = dif_cot_w

            half_dif_cot_u = dif_cot_u / 2.
            half_dif_cot_v = dif_cot_v / 2.
            half_dif_cot_w = dif_cot_w / 2.

            # Set dif_LC
            edge = halfedge.edge
            self.dif_LC_edges[edge.index][u.index] += half_dif_cot_u
            self.dif_LC_edges[edge.index][v.index] += half_dif_cot_v
            self.dif_LC_edges[edge.index][w.index] += half_dif_cot_w
            self.dif_LC_vertices[u.index][u.index] -= half_dif_cot_u
            self.dif_LC_vertices[u.index][v.index] -= half_dif_cot_v
            self.dif_LC_vertices[u.index][w.index] -= half_dif_cot_w
            self.dif_LC_vertices[v.index][u.index] -= half_dif_cot_u
            self.dif_LC_vertices[v.index][v.index] -= half_dif_cot_v
            self.dif_LC_vertices[v.index][w.index] -= half_dif_cot_w

            # Set dif_LC_interior
            if not u.is_on_boundary() and not v.is_on_boundary():
                self.dif_LC_interior_edges[edge.index][u.index] += half_dif_cot_u
                self.dif_LC_interior_edges[edge.index][v.index] += half_dif_cot_v
                self.dif_LC_interior_edges[edge.index][w.index] += half_dif_cot_w
                self.dif_LC_interior_vertices[u.index][u.index] -= half_dif_cot_u
                self.dif_LC_interior_vertices[u.index][v.index] -= half_dif_cot_v
                self.dif_LC_interior_vertices[u.index][w.index] -= half_dif_cot_w
                self.dif_LC_interior_vertices[v.index][u.index] -= half_dif_cot_u
                self.dif_LC_interior_vertices[v.index][v.index] -= half_dif_cot_v
                self.dif_LC_interior_vertices[v.index][w.index] -= half_dif_cot_w
