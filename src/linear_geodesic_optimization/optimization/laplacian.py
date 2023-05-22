"""Module containing utilities to compute the Laplace-Beltrami operator."""

import itertools
import typing

import dcelmesh
import numpy as np
import numpy.typing as npt

from linear_geodesic_optimization.mesh.mesh import Mesh


class Computer:
    """Implementation of the Laplace-Beltrami operator on a mesh."""

    def __init__(self, mesh: Mesh):
        """Initialize the computer."""
        self._mesh: Mesh = mesh
        self._topology: dcelmesh.Mesh = mesh.get_topology()

        # Forward variables
        self._forward_updates: int = mesh.get_updates() - 1
        self._coordinates: npt.NDArray[np.float64] \
            = np.zeros((self._topology.n_vertices(), 3))
        self.N: typing.List[npt.NDArray[np.float64]] \
            = [np.zeros(3) for _ in self._topology.faces()]
        """A list of normals of faces, indexed by faces."""
        self.A: typing.List[np.float64] \
            = [np.float64(0.) for _ in self._topology.faces()]
        """A list of areas of faces, indexed by faces."""
        self.D: typing.List[np.float64] \
            = [np.float64(0.) for _ in self._topology.vertices()]
        """A list of vertex areas, indexed by vertices."""
        self.cot: typing.List[np.float64] \
            = [np.float64(0.) for _ in self._topology.halfedges()]
        """
        A list of cotangents of the opposing angles to halfedges,
        indexed by halfedges.
        """
        self.LC_neumann_edges: typing.List[np.float64] \
            = [np.float64(0.) for _ in self._topology.edges()]
        """
        A list of (non-trivial) off-diagonal entries of the
        Laplace-Beltrami operator with Neumann boundary conditions,
        indexed by edges.
        """
        self.LC_neumann_vertices: typing.List[np.float64] \
            = [np.float64(0.) for _ in self._topology.vertices()]
        """
        A list of diagonal entries of the Laplace-Beltrami operator with
        Neumann boundary conditions, indexed by vertices.
        """
        self.LC_dirichlet_edges: typing.List[np.float64] \
            = [np.float64(0.) for _ in self._topology.edges()]
        """
        A list of (non-trivial) off-diagonal entries of the
        Laplace-Beltrami operator with Dirichlet boundary conditions,
        indexed by edges.
        """
        self.LC_dirichlet_vertices: typing.List[np.float64] \
            = [np.float64(0.) for _ in self._topology.vertices()]
        """
        A list of diagonal entries of the Laplace-Beltrami operator with
        Dirichlet boundary conditions, indexed by vertices.
        """

        # Reverse variables
        self._reverse_updates: int = mesh.get_updates() - 1
        self._partials: npt.NDArray[np.float64] \
            = np.zeros((self._topology.n_vertices(), 3))
        self.dif_N: typing.List[typing.Dict[int, npt.NDArray[np.float64]]] \
            = [{} for _ in self._topology.faces()]
        """
        A list of partials of normals of faces, indexed by faces and
        then by (incident) vertices.
        """
        self.dif_A: typing.List[typing.Dict[int, np.float64]] \
            = [{} for _ in self._topology.faces()]
        """
        A list partials of areas of faces, indexed by faces and then by
        (incident) vertices.
        """
        self.dif_D: typing.List[typing.Dict[int, np.float64]] \
            = [{} for _ in self._topology.vertices()]
        """
        A list of partials of vertex areas, indexed by vertices and
        then by vertices (at most distance 1 away).
        """
        self.dif_cot: typing.List[typing.Dict[int, np.float64]] \
            = [{} for _ in self._topology.halfedges()]
        """
        A list of partials of cotangents of the opposing angles to
        halfedges, indexed by halfedges and then by vertices (of the
        same face).
        """
        self.dif_LC_neumann_edges: \
            typing.List[typing.Dict[int, np.float64]] \
            = [{} for _ in self._topology.edges()]
        """
        A list of partials of (non-trivial) off-diagonal entries of the
        Laplace-Beltrami operator with Neumann boundary conditions,
        indexed by edges and then by vertices (of the same and opposing
        faces).
        """
        self.dif_LC_neumann_vertices: \
            typing.List[typing.Dict[int, np.float64]] \
            = [{} for _ in self._topology.vertices()]
        """
        A list of partials of diagonal entries of the Laplace-Beltrami
        operator with Neumann boundary conditions, indexed by vertices
        and then vertices (at most distance 1 away).
        """
        self.dif_LC_dirichlet_edges: \
            typing.List[typing.Dict[int, np.float64]] \
            = [{} for _ in self._topology.edges()]
        """
        A list of partials of (non-trivial) off-diagonal entries of the
        Laplace-Beltrami operator with Dirichlet boundary conditions,
        indexed by edges and then by vertices (of the same and opposing
        faces).
        """
        self.dif_LC_dirichlet_vertices: \
            typing.List[typing.Dict[int, np.float64]] \
            = [{} for _ in self._topology.vertices()]
        """
        A list of partials of diagonal entries of the Laplace-Beltrami
        operator with Dirichlet boundary conditions, indexed by vertices
        and then vertices (at most distance 1 away).
        """

    def forward(self) -> None:
        """
        Compute the forward direction.

        The computed values will be stored in the variables:
        * `Computer.N`
        * `Computer.A`
        * `Computer.D`
        * `Computer.cot`
        * `Computer.LC_neumann_halfedges`
        * `Computer.LC_neumann_vertices`
        * `Computer.LC_dirichlet_halfedges`
        * `Computer.LC_dirichlet_vertices`
        """
        if self._forward_updates == self._mesh.get_updates():
            return
        self._forward_updates = self._mesh.get_updates()
        self._coordinates = self._mesh.get_coordinates()

        # Reset quantities that will be computed via accumulation
        self.D = [np.float64(0.) for _ in self._topology.vertices()]
        self.LC_neumann_edges \
            = [np.float64(0.) for _ in self._topology.edges()]
        self.LC_neumann_vertices \
            = [np.float64(0.) for _ in self._topology.vertices()]
        self.LC_dirichlet_edges \
            = [np.float64(0.) for _ in self._topology.edges()]
        self.LC_dirichlet_vertices \
            = [np.float64(0.) for _ in self._topology.vertices()]

        # N and A can be computed by iterating over faces
        for face in self._topology.faces():
            u, v, w = face.vertices()
            pu = self._coordinates[u.index()]
            pv = self._coordinates[v.index()]
            pw = self._coordinates[w.index()]

            # Set N
            normal = np.cross(pu - pw, pv - pw)
            self.N[face.index()] = normal

            # Set A
            area = np.linalg.norm(normal) / 2.
            self.A[face.index()] = area

        # D, cot, and L_C can be computed by iterating over halfedges
        for halfedge in self._topology.halfedges():
            u = halfedge.origin()
            v = halfedge.destination()
            w = halfedge.previous().origin()
            pu = self._coordinates[u.index()]
            pv = self._coordinates[v.index()]
            pw = self._coordinates[w.index()]

            area = self.A[halfedge.face().index()]

            # Set D
            self.D[u.index()] += area / 3.

            # Set cot
            cotangent = (pu - pw) @ (pv - pw) / (2. * area)
            self.cot[halfedge.index()] = cotangent

            half_cotangent = cotangent / 2.

            # Set LC_neumann
            edge = halfedge.edge()
            self.LC_neumann_edges[edge.index()] += half_cotangent
            self.LC_neumann_vertices[u.index()] -= half_cotangent
            self.LC_neumann_vertices[v.index()] -= half_cotangent

            # Set LC_dirichlet
            if not u.is_on_boundary() and not v.is_on_boundary():
                self.LC_dirichlet_edges[edge.index()] += half_cotangent
                self.LC_dirichlet_vertices[u.index()] -= half_cotangent
                self.LC_dirichlet_vertices[v.index()] -= half_cotangent

    def reverse(self) -> None:
        """
        Compute the reverse direction (that is, partials).

        The computed values will be stored in the variables:
        * `Computer.dif_N`
        * `Computer.dif_A`
        * `Computer.dif_D`
        * `Computer.dif_cot`
        * `Computer.dif_LC_neumann_halfedges`
        * `Computer.dif_LC_neumann_vertices`
        * `Computer.dif_LC_dirichlet_halfedges`
        * `Computer.dif_LC_dirichlet_vertices`
        """
        if self._reverse_updates == self._mesh.get_updates():
            return
        self.forward()
        self._reverse_updates = self._mesh.get_updates()
        self._partials = self._mesh.get_partials()

        # Reset quantities that will be computed via accumulation
        self.dif_D = [
            {
                near.index(): np.float64(0.)
                for near in itertools.chain([vertex], vertex.vertices())
            }
            for vertex in self._topology.vertices()
        ]
        self.dif_LC_neumann_edges = [
            {
                vertex.index(): np.float64(0.)
                for face in edge.faces()
                for vertex in face.vertices()
            }
            for edge in self._topology.edges()
        ]
        self.dif_LC_neumann_vertices = [
            {
                near.index(): np.float64(0.)
                for near in itertools.chain([vertex], vertex.vertices())
            }
            for vertex in self._topology.vertices()
        ]
        self.dif_LC_dirichlet_edges = [
            {
                vertex.index(): np.float64(0.)
                for face in edge.faces()
                for vertex in face.vertices()
            }
            for edge in self._topology.edges()
        ]
        self.dif_LC_dirichlet_vertices = [
            {
                near.index(): np.float64(0.)
                for near in itertools.chain([vertex], vertex.vertices())
            }
            for vertex in self._topology.vertices()
        ]

        for halfedge in self._topology.halfedges():
            u = halfedge.origin()
            v = halfedge.destination()
            w = halfedge.previous().origin()
            pu = self._coordinates[u.index()]
            pv = self._coordinates[v.index()]
            pw = self._coordinates[w.index()]

            # Set dif_N
            normal = self.N[halfedge.face().index()]
            dif_N_u = np.cross(pw - pv, self._partials[u.index()])
            self.dif_N[halfedge.face().index()][u.index()] = dif_N_u

            # Set dif_A
            area = self.A[halfedge.face().index()]
            dif_A_u = normal @ (dif_N_u) / (4. * area)
            self.dif_A[halfedge.face().index()][u.index()] = dif_A_u

            # Set dif_D
            third_dif_A_u = dif_A_u / 3.
            self.dif_D[u.index()][u.index()] += third_dif_A_u
            self.dif_D[v.index()][u.index()] += third_dif_A_u
            self.dif_D[w.index()][u.index()] += third_dif_A_u

        # Need a separate loop here to ensure dif_A has been computed
        for halfedge in self._topology.halfedges():
            u = halfedge.origin()
            v = halfedge.destination()
            w = halfedge.previous().origin()
            pu = self._coordinates[u.index()]
            pv = self._coordinates[v.index()]
            pw = self._coordinates[w.index()]

            # Set dif_cot
            area = self.A[halfedge.face().index()]
            cotangent = self.cot[halfedge.index()]
            dif_A = self.dif_A[halfedge.face().index()]
            dif_cot_u = ((pv - pw) @ self._partials[u.index()]
                         - 2. * cotangent * dif_A[u.index()]) / (2. * area)
            dif_cot_v = ((pu - pw) @ self._partials[v.index()]
                         - 2. * cotangent * dif_A[v.index()]) / (2. * area)
            dif_cot_w = ((2. * pw - pu - pv) @ self._partials[w.index()]
                         - 2. * cotangent * dif_A[w.index()]) / (2. * area)
            self.dif_cot[halfedge.index()][u.index()] = dif_cot_u
            self.dif_cot[halfedge.index()][v.index()] = dif_cot_v
            self.dif_cot[halfedge.index()][w.index()] = dif_cot_w

            half_dif_cot_u = dif_cot_u / 2.
            half_dif_cot_v = dif_cot_v / 2.
            half_dif_cot_w = dif_cot_w / 2.

            # Set dif_LC_neumann
            edge = halfedge.edge()
            self.dif_LC_neumann_edges[edge.index()][u.index()] \
                += half_dif_cot_u
            self.dif_LC_neumann_edges[edge.index()][v.index()] \
                += half_dif_cot_v
            self.dif_LC_neumann_edges[edge.index()][w.index()] \
                += half_dif_cot_w
            self.dif_LC_neumann_vertices[u.index()][u.index()] \
                -= half_dif_cot_u
            self.dif_LC_neumann_vertices[u.index()][v.index()] \
                -= half_dif_cot_v
            self.dif_LC_neumann_vertices[u.index()][w.index()] \
                -= half_dif_cot_w
            self.dif_LC_neumann_vertices[v.index()][u.index()] \
                -= half_dif_cot_u
            self.dif_LC_neumann_vertices[v.index()][v.index()] \
                -= half_dif_cot_v
            self.dif_LC_neumann_vertices[v.index()][w.index()] \
                -= half_dif_cot_w

            # Set dif_LC_dirichlet
            if not u.is_on_boundary() and not v.is_on_boundary():
                self.dif_LC_dirichlet_edges[edge.index()][u.index()] \
                    += half_dif_cot_u
                self.dif_LC_dirichlet_edges[edge.index()][v.index()] \
                    += half_dif_cot_v
                self.dif_LC_dirichlet_edges[edge.index()][w.index()] \
                    += half_dif_cot_w
                self.dif_LC_dirichlet_vertices[u.index()][u.index()] \
                    -= half_dif_cot_u
                self.dif_LC_dirichlet_vertices[u.index()][v.index()] \
                    -= half_dif_cot_v
                self.dif_LC_dirichlet_vertices[u.index()][w.index()] \
                    -= half_dif_cot_w
                self.dif_LC_dirichlet_vertices[v.index()][u.index()] \
                    -= half_dif_cot_u
                self.dif_LC_dirichlet_vertices[v.index()][v.index()] \
                    -= half_dif_cot_v
                self.dif_LC_dirichlet_vertices[v.index()][w.index()] \
                    -= half_dif_cot_w
