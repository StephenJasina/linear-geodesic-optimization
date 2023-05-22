"""Module containing utilities to compute curvature on a mesh."""

import itertools
import typing

import dcelmesh
import numpy as np
import numpy.typing as npt

from linear_geodesic_optimization.mesh.mesh import Mesh
from linear_geodesic_optimization.optimization.laplacian \
    import Computer as Laplacian


class Computer:
    """Implementation of curvature operators on a mesh."""

    def __init__(self, mesh: Mesh, laplacian: Laplacian):
        """Initialize the computer."""
        self._mesh = mesh
        self._topology = mesh.get_topology()
        self._laplacian = laplacian

        # Forward variables
        self._forward_updates: int = mesh.get_updates() - 1
        self._coordinates: npt.NDArray[np.float64] \
            = np.zeros((self._topology.n_vertices(), 3))
        self.vertex_N: typing.List[npt.NDArray[np.float64]] \
            = [np.zeros(3) for _ in self._topology.vertices()]
        """
        A list of approximate normals at vertices, indexed by vertices.
        """
        self.mean_curvature_normal: typing.List[npt.NDArray[np.float64]] \
            = [np.zeros(3) for _ in self._topology.vertices()]
        """A list of mean curvature normals, indexed by vertices."""
        self.kappa_G: typing.List[np.float64] \
            = [np.float64(0.) for _ in self._topology.vertices()]
        """A list of Gaussian curvatures, indexed by vertices."""
        self.kappa_H: typing.List[np.float64] \
            = [np.float64(0.) for _ in self._topology.vertices()]
        """A list of mean curvatures, indexed by vertices."""
        self.kappa_1: typing.List[np.float64] \
            = [np.float64(0.) for _ in self._topology.vertices()]
        """A list of first principal curvatures, indexed by vertices."""
        self.kappa_2: typing.List[np.float64] \
            = [np.float64(0.) for _ in self._topology.vertices()]
        """A list of second principal curvatures, indexed by vertices."""

        # Reverse variables
        self._reverse_updates: int = mesh.get_updates() - 1
        self._partials: npt.NDArray[np.float64] \
            = np.zeros((self._topology.n_vertices(), 3))
        self.dif_mean_curvature_normal: \
            typing.List[typing.Dict[int, npt.NDArray[np.float64]]] \
            = [{} for _ in self._topology.vertices()]
        """
        A list of partials of mean curvature normals, indexed by
        vertices, and then by vertices (at most distance 1 away).
        """
        self.dif_kappa_G: typing.List[typing.Dict[int, np.float64]] \
            = [{} for _ in self._topology.vertices()]
        """
        A list of partials of Gaussian curvatures, indexed by vertices,
        and then by vertices (at most distance 1 away).
        """
        self.dif_kappa_H: typing.List[typing.Dict[int, np.float64]] \
            = [{} for _ in self._topology.vertices()]
        """
        A list of partials of mean curvatures, indexed by vertices, and
        then by vertices (at most distance 1 away).
        """
        self.dif_kappa_1: typing.List[typing.Dict[int, np.float64]] \
            = [{} for _ in self._topology.vertices()]
        """
        A list of partials of first principal curvatures, indexed by
        vertices, and then by vertices (at most distance 1 away).
        """
        self.dif_kappa_2: typing.List[typing.Dict[int, np.float64]] \
            = [{} for _ in self._topology.vertices()]
        """
        A list of partials of second principal curvatures, indexed by
        vertices, and then by vertices (at most distance 1 away).
        """

    def forward(self) -> None:
        """
        Compute the forward direction.

        The computed values will be stored in the variables:
        * `Computer.vertex_N`
        * `Computer.mean_curvature_normal`
        * `Computer.kappa_G`
        * `Computer.kappa_H`
        * `Computer.kappa_1`
        * `Computer.kappa_2`
        """
        if self._forward_updates == self._mesh.get_updates():
            return
        self._laplacian.forward()
        self._forward_updates = self._mesh.get_updates()
        self._coordinates = self._mesh.get_coordinates()

        # Compute Gaussian curvatures
        self.kappa_G = [
            np.float64(0.)
            if vertex.is_on_boundary()
            else np.float64(2. * np.pi) / self._laplacian.D[vertex.index()]
            for vertex in self._topology.vertices()
        ]
        for halfedge in self._topology.halfedges():
            w = halfedge.previous().origin()
            if w.is_on_boundary():
                continue
            cotangent = self._laplacian.cot[halfedge.index()]
            self.kappa_G[w.index()] -= np.arccos(
                cotangent / (1 + cotangent**2)**0.5
            ) / self._laplacian.D[w.index()]

        # Compute vertex normals
        self.vertex_N = [np.zeros(3) for _ in self._topology.vertices()]
        for face in self._topology.faces():
            face_normal = self._laplacian.N[face.index()]
            for vertex in face.vertices():
                self.vertex_N[vertex.index()] += face_normal

        # Compute mean curvature normals
        self.mean_curvature_normal \
            = [np.zeros(3) for _ in self._topology.vertices()]
        for edge in self._topology.edges():
            u, v = edge.vertices()
            laplacian = self._laplacian.LC_neumann_edges[edge.index()]
            self.mean_curvature_normal[u.index()] \
                -= laplacian * self._coordinates[v.index()] \
                / (2. * self._laplacian.D[u.index()])
            self.mean_curvature_normal[v.index()] \
                -= laplacian * self._coordinates[u.index()] \
                / (2. * self._laplacian.D[v.index()])
        for vertex in self._topology.vertices():
            self.mean_curvature_normal[vertex.index()] \
                -= self._laplacian.LC_neumann_vertices[vertex.index()] \
                * self._coordinates[vertex.index()] \
                / (2. * self._laplacian.D[vertex.index()])

        # Compute mean and principal curvatures
        for vertex in self._topology.vertices():
            mean_curvature_normal = self.mean_curvature_normal[vertex.index()]
            vertex_normal = self.vertex_N[vertex.index()]
            kappa_G = self.kappa_G[vertex.index()]
            if vertex.is_on_boundary():
                kappa_H = np.float64(0.)
            else:
                kappa_H = np.linalg.norm(mean_curvature_normal) \
                    * np.sign(mean_curvature_normal @ vertex_normal)
            self.kappa_H[vertex.index()] = kappa_H
            offset = np.sqrt(np.maximum(0., kappa_H**2 - kappa_G))
            self.kappa_1[vertex.index()] \
                = kappa_H + offset
            self.kappa_2[vertex.index()] \
                = kappa_H - offset

    def reverse(self) -> None:
        """
        Compute the reverse direction (that is, partials).

        The computed values will be stored in the variables:
        * `Computer.dif_mean_curvature_normal`
        * `Computer.dif_kappa_G`
        * `Computer.dif_kappa_H`
        * `Computer.dif_kappa_1`
        * `Computer.dif_kappa_2`
        """
        if self._reverse_updates == self._mesh.get_updates():
            return
        self.forward()
        self._laplacian.reverse()
        self._reverse_updates = self._mesh.get_updates()
        self._partials = self._mesh.get_partials()

        # Compute partials of Gaussian curvatures
        self.dif_kappa_G = [
            {}
            if vertex.is_on_boundary()
            else {
                near.index():
                -self._laplacian.dif_D[vertex.index()][near.index()]
                * self.kappa_G[vertex.index()]
                / self._laplacian.D[vertex.index()]
                for near in itertools.chain([vertex], vertex.vertices())
            }
            for vertex in self._topology.vertices()
        ]
        for halfedge in self._topology.halfedges():
            u = halfedge.origin()
            v = halfedge.destination()
            w = halfedge.previous().origin()
            if w.is_on_boundary():
                continue
            for vertex in [u, v, w]:
                self.dif_kappa_G[w.index()][vertex.index()] \
                    += self._laplacian.dif_cot[halfedge.index()][vertex.index()] \
                    / (1 + self._laplacian.cot[halfedge.index()]**2) \
                    / self._laplacian.D[w.index()]

        # Compute partials of mean curvature normals
        self.dif_mean_curvature_normal = [
            {
                near_near.index(): np.zeros(3)
                for near in vertex.vertices()
                for near_near in near.vertices()
            }
            for vertex in self._topology.vertices()
        ]
        for edge in self._topology.edges():
            w, x = edge.vertices()
            laplacian = self._laplacian.LC_neumann_edges[edge.index()]

            for u, v in [(w, x), (x, w)]:
                vertex_area = self._laplacian.D[u.index()]
                self.dif_mean_curvature_normal[u.index()][v.index()] \
                    -= self._laplacian.LC_neumann_edges[edge.index()] \
                    * self._partials[v.index()] \
                    / (2. * vertex_area)
                for near in itertools.chain([u], u.vertices()):
                    dif_vertex_area \
                        = self._laplacian.dif_D[u.index()][near.index()]
                    if near.index() \
                            in self._laplacian.dif_LC_neumann_edges[edge.index()]:
                        dif_laplacian \
                            = self._laplacian.dif_LC_neumann_edges[edge.index()][near.index()]
                    else:
                        dif_laplacian = np.float64(0.)
                    self.dif_mean_curvature_normal[u.index()][near.index()] \
                        -= (dif_laplacian
                            - dif_vertex_area * laplacian / vertex_area) \
                        * self._coordinates[v.index()] \
                        / (2. * vertex_area)
        for vertex in self._topology.vertices():
            coordinates = self._coordinates[vertex.index()]
            vertex_area = self._laplacian.D[vertex.index()]
            laplacian = self._laplacian.LC_neumann_vertices[vertex.index()]
            self.dif_mean_curvature_normal[vertex.index()][vertex.index()] \
                -= self._laplacian.LC_neumann_vertices[vertex.index()] \
                * self._partials[vertex.index()] \
                / (2. * vertex_area)
            for near in itertools.chain([vertex], vertex.vertices()):
                dif_vertex_area = self._laplacian.dif_D[vertex.index()][near.index()]
                dif_laplacian \
                    = self._laplacian.dif_LC_neumann_vertices[vertex.index()][near.index()]
                self.dif_mean_curvature_normal[vertex.index()][near.index()] \
                    -= ((dif_laplacian
                         - dif_vertex_area * laplacian / vertex_area)
                        * coordinates) \
                    / (2. * vertex_area)

        # Compute partials of mean and principal curvatures
        for vertex in self._topology.vertices():
            if vertex.is_on_boundary():
                continue
            mean_curvature_normal = self.mean_curvature_normal[vertex.index()]
            vertex_normal = self.vertex_N[vertex.index()]
            kappa_1 = self.kappa_1[vertex.index()]
            kappa_2 = self.kappa_2[vertex.index()]
            curvature_difference = kappa_1 - kappa_2
            for near in itertools.chain([vertex], vertex.vertices()):
                dif_kappa_G = self.dif_kappa_G[vertex.index()][near.index()]
                dif_mean_curvature_normal \
                    = self.dif_mean_curvature_normal[vertex.index()][near.index()]
                dif_kappa_H = np.sign(vertex_normal @ mean_curvature_normal) \
                    * (mean_curvature_normal @ dif_mean_curvature_normal) \
                    / np.linalg.norm(mean_curvature_normal)
                self.dif_kappa_H[vertex.index()][near.index()] = dif_kappa_H
                if curvature_difference == np.float64(0.):
                    self.dif_kappa_1[vertex.index()][near.index()] \
                        = np.float64(0.)
                    self.dif_kappa_2[vertex.index()][near.index()] \
                        = np.float64(0.)
                else:
                    self.dif_kappa_1[vertex.index()][near.index()] \
                        = (2. * kappa_1 * dif_kappa_H - dif_kappa_G) \
                        / curvature_difference
                    self.dif_kappa_2[vertex.index()][near.index()] \
                        = (dif_kappa_G - 2. * kappa_2 * dif_kappa_H) \
                        / curvature_difference
