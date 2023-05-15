"""Module containing utilities to compute the Laplace-Beltrami operator."""

import typing

import dcelmesh
import numpy as np
import numpy.typing as npt

from linear_geodesic_optimization.mesh.mesh import Mesh


class Computer:
    """
    Implementation of the Laplace-Beltrami operator on a mesh.
    """

    def __init__(self, mesh: Mesh):
        """Initialize the computer."""
        self._mesh: Mesh = mesh
        self._topology: dcelmesh.Mesh = mesh.get_topology()
        self._coordinates: typing.Optional[npt.NDArray[np.float64]] = None

        self._updates: int = mesh.get_updates() - 1

        # An array of normals of faces
        self.N: npt.NDArray[np.float64] \
            = np.zeros((self._topology.n_faces(), 3))

        # An array of areas of faces
        self.A: npt.NDArray[np.float64] = np.zeros(self._topology.n_faces())

        # An array of vertex areas
        self.D: npt.NDArray[np.float64] = np.zeros(self._topology.n_vertices())

        # An array of cotangents of the opposing angles to halfedges
        self.cot: npt.NDArray[np.float64] \
            = np.zeros(3 * self._topology.n_faces())

        # Lists of (non-zero) entries of the Laplace-Beltrami operator
        # with Neumann boundary conditions.
        self.LC_neumann_halfedges: npt.NDArray[np.float64] \
            = np.zeros(self._topology.n_halfedges())
        self.LC_neumann_vertices: npt.NDArray[np.float64] \
            = np.zeros(self._topology.n_vertices())

        # Lists of (non-zero) entries of the Laplace-Beltrami operator
        # with Dirichlet boundary conditions.
        self.LC_dirichlet_halfedges: npt.NDArray[np.float64] \
            = np.zeros(self._topology.n_halfedges())
        self.LC_dirichlet_vertices: npt.NDArray[np.float64] \
            = np.zeros(self._topology.n_vertices())

    def forward(self) -> typing.NoReturn:
        """
        Compute the forward direction.

        The computed values will be stored in the variables:
        * Computer.N
        * Computer.A
        * Computer.D
        * Computer.cot
        * Computer.LC_neumann
        * Computer.LC_dirichlet
        """
        if self._updates == self._mesh.get_updates():
            return

        self._coordinates = self._mesh.get_coordinates()

        self.D = np.zeros(self._topology.n_vertices())
        for face in self._topology.faces():
            u, v, w = face.vertices()
            pu = self._coordinates[u.index()]
            pv = self._coordinates[v.index()]
            pw = self._coordinates[w.index()]

            # Set N
            normal = np.cross(pu - pw, pv - pw)
            self.N[face.index(), :] = normal

            # Set A
            area = np.linalg.norm(normal) / 2.
            self.A[face.index()] = area

            # Set D
            third_area = area / 3.
            self.D[u.index()] += third_area
            self.D[v.index()] += third_area
            self.D[w.index()] += third_area

        self.LC_neumann_halfedges = np.zeros(self._topology.n_halfedges())
        self.LC_neumann_vertices = np.zeros(self._topology.n_vertices())
        self.LC_dirichlet_halfedges = np.zeros(self._topology.n_halfedges())
        self.LC_dirichlet_vertices = np.zeros(self._topology.n_vertices())
        for halfedge in self._topology.halfedges():
            u = halfedge.origin()
            v = halfedge.destination()
            w = halfedge.previous().origin()
            pu = self._coordinates[u.index()]
            pv = self._coordinates[v.index()]
            pw = self._coordinates[w.index()]

            # Set cot
            cotangent = (pu - pw) @ (pv - pw) / (2. * area)
            self.cot[halfedge.index()] = cotangent

            half_cotangent = cotangent / 2.

            # Set LC_neumann
            self.LC_neumann_halfedges[halfedge.index()] += half_cotangent
            if not halfedge.is_on_boundary():
                self.LC_neumann_halfedges[halfedge.twin().index()] \
                    += half_cotangent
            self.LC_neumann_vertices[halfedge.origin().index()] \
                -= half_cotangent
            self.LC_neumann_vertices[halfedge.destination().index()] \
                -= half_cotangent

            # Set LC_dirichlet
            if not (halfedge.origin().is_on_boundary()
                    or halfedge.destination().is_on_boundary()):
                self.LC_dirichlet_halfedges[halfedge.index()] += half_cotangent
                self.LC_dirichlet_halfedges[halfedge.twin().index()] \
                    += half_cotangent
                self.LC_dirichlet_vertices[halfedge.origin().index()] \
                    -= half_cotangent
                self.LC_dirichlet_vertices[halfedge.destination().index()] \
                    -= half_cotangent
