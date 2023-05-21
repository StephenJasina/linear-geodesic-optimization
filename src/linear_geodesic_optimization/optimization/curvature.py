"""Module containing utilities to compute curvature on a mesh."""

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
        self.dif_vertex_N: \
            typing.List[typing.Dict[int, npt.NDArray[np.float64]]] \
            = [{} for _ in self._topology.vertices()]
        """
        A list of partials of approximate normals at vertices, indexed
        by vertices, and then by vertices (at most distance 1 away).
        """
        self.dif_mean_curvature_normal: \
            typing.List[typing.Dict[int, npt.NDArray[np.float64]]] \
            = [{} for _ in self._topology.vertices()]
        """
        A list of partials of mean curvature normals, indexed by
        vertices, and then by vertices (at most distance 1 away).
        """
        self.dif_kappa_G: typing.List[typing.Dict[int, np.float64]] \
            = [np.float64(0.) for _ in self._topology.vertices()]
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

        # Start to compute Gaussian curvatures
        self.kappa_G = [
            np.float64(2. * np.pi)
            for _ in self._topology.vertices()
        ]
        for halfedge in self._topology.halfedges():
            w = halfedge.previous().origin()
            # if w.is_on_boundary():
            #     continue
            cotangent = self._laplacian.cot[halfedge.index()]
            self.kappa_G[w.index()] -= np.arccos(
                cotangent / (1 + cotangent**2)**0.5
            )

        # Compute vertex normals
        self.vertex_N = [np.zeros(3) for _ in self._topology.vertices()]
        for face in self._topology.faces():
            face_normal = self._laplacian.N[face.index()]
            for vertex in face.vertices():
                self.vertex_N[vertex.index()] += face_normal

        # Start to compute mean curvature normals
        self.mean_curvature_normal \
            = [np.zeros(3) for _ in self._topology.vertices()]
        for edge in self._topology.edges():
            u, v = edge.vertices()
            laplacian_element \
                = self._laplacian.LC_neumann_halfedges[edge.index()]
            self.mean_curvature_normal[u.index()] \
                += laplacian_element * self._coordinates[v.index()]
            self.mean_curvature_normal[v.index()] \
                += laplacian_element * self._coordinates[u.index()]
        for vertex in self._topology.vertices():
            self.mean_curvature_normal[vertex.index()] \
                += self._laplacian.LC_neumann_vertices[vertex.index()] \
                * self._coordinates[vertex.index()]

        # Finish computing Gaussian curvatures and mean curvature
        # normals. Also compute mean and principal curvatures.
        for vertex in self._topology.vertices():
            index = vertex.index()
            vertex_area = self._laplacian.D[index]
            self.kappa_G[index] /= vertex_area
            self.mean_curvature_normal[index] /= -2 * vertex_area
            mean_curvature_normal = self.mean_curvature_normal[index]
            vertex_normal = self.vertex_N[index]
            self.kappa_H[index] \
                = np.linalg.norm(mean_curvature_normal) \
                * np.sign(mean_curvature_normal @ vertex_normal)
            if vertex.is_on_boundary():
                self.kappa_G[index] = np.float64(0.)
                self.kappa_H[index] = np.float64(0.)
            kappa_G = self.kappa_G[index]
            kappa_H = self.kappa_H[index]
            self.kappa_1[index] \
                = kappa_H + np.sqrt(np.maximum(0., kappa_H**2 - kappa_G))
            self.kappa_2[index] \
                = kappa_H - np.sqrt(np.maximum(0., kappa_H**2 - kappa_G))

    def reverse(self) -> None:
        """
        Compute the reverse direction (that is, partials).

        The computed values will be stored in the variables:
        * `Computer.dif_vertex_N`
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


# TODO: Replace this
class Reverse:
    '''
    Implementation of the gradient of the curvature loss function on a mesh.
    This implementation assumes the l-th partial affects only the l-th vertex.
    '''

    def __init__(self, mesh, laplacian_forward=None, curvature_forward=None,
                 laplacian_reverse=None):
        self._mesh = mesh
        self._updates = self._mesh.updates() - 1
        self._v = None
        self._e = self._mesh.get_edges()
        self._nxt = self._mesh.get_nxt()

        self._V = len(self._e)

        self._dif_v = None
        self._l = None

        self._laplacian_forward = laplacian_forward
        if self._laplacian_forward is None:
            self._laplacian_forward = laplacian.Forward(mesh)

        self._curvature_forward = curvature_forward
        if self._curvature_forward is None:
            self._curvature_forward = Forward(mesh, self._laplacian_forward)

        self._laplacian_reverse = laplacian_reverse
        if self._laplacian_reverse is None:
            self._laplacian_reverse = laplacian.Reverse(mesh)

        self._N = None
        self._D_inv = None
        self._cot = None
        self._LC = None

        self._kappa_G = None
        self._vertex_N = None
        self._mean_curvature_normal = None

        self._dif_N = None
        self._dif_D = None
        self._dif_cot = None
        self._dif_LC = None

        # Derivatives match the types of what are being differentiated.
        self.dif_kappa_G = None
        self.dif_vertex_N = None
        self.dif_mean_curvature_normal = None
        self.dif_kappa_H = None
        self.dif_kappa_1 = None
        self.dif_kappa_2 = None

    def _calc_dif_kappa_G(self):
        dif_Dkappa_G = np.zeros(self._V)
        for (i, j), cot_ij in self._cot.items():
            if (i, j) in self._dif_cot:
                dif_Dkappa_G[self._nxt[i,j]] += self._dif_cot[i,j] / (1 + cot_ij**2)
        return self._D_inv @ (dif_Dkappa_G - self._dif_D @ self._kappa_G)

    def _calc_dif_vertex_N(self):
        dif_vertex_normal = np.zeros((self._V, 3))
        for i in range(self._V):
            for j in self._e[i]:
                dif_N = self._dif_N[i, j] if (i, j) in self._dif_N else 0.
                dif_vertex_normal[i,:] += dif_N
        return dif_vertex_normal

    def _calc_dif_mean_curvature_normal(self):
        dif_v = np.zeros((self._V, 3))
        dif_v[self._l,:] = self._dif_v
        return -self._D_inv @ (
            (self._dif_LC - self._dif_D @ self._D_inv @ self._LC) @ self._v
            + self._LC @ dif_v
        ) / 2

    def _calc_dif_kappa_H(self):
        dif_kappa_H = np.zeros(self._V)
        for i in range(self._V):
            vn = self._vertex_N[i,:]
            mcn = self._mean_curvature_normal[i,:]
            dif_mcn = self.dif_mean_curvature_normal[i,:]
            dif_kappa_H[i] = np.sign(vn @ mcn) * (mcn @ dif_mcn) / np.linalg.norm(mcn)
        return dif_kappa_H

    def _calc_dif_kappa_1(self):
        return np.divide(
            2 * self._kappa_1 * self.dif_kappa_H - self.dif_kappa_G,
            self._kappa_1 - self._kappa_2,
            np.copy(self.dif_kappa_H),
            where=(self._kappa_1 != self._kappa_2)
        )

    def _calc_dif_kappa_2(self):
        return np.divide(
            self.dif_kappa_G - 2 * self._kappa_2 * self.dif_kappa_H,
            self._kappa_1 - self._kappa_2,
            np.copy(self.dif_kappa_H),
            where=(self._kappa_1 != self._kappa_2)
        )

    def calc(self, dif_v, l):
        self._laplacian_forward.calc()
        self._N = self._laplacian_forward.N
        self._D_inv = self._laplacian_forward.D_inv
        self._cot = self._laplacian_forward.cot
        self._LC = self._laplacian_forward.LC_neumann

        self._curvature_forward.calc()
        self._kappa_G = self._curvature_forward.kappa_G
        self._vertex_N = self._curvature_forward.vertex_N
        self._mean_curvature_normal = self._curvature_forward.mean_curvature_normal
        self._kappa_H = self._curvature_forward.kappa_H
        self._kappa_1 = self._curvature_forward.kappa_1
        self._kappa_2 = self._curvature_forward.kappa_2

        self._laplacian_reverse.calc(dif_v, l)
        self._dif_N = self._laplacian_reverse.dif_N
        self._dif_D = self._laplacian_reverse.dif_D
        self._dif_cot = self._laplacian_reverse.dif_cot
        self._dif_LC = self._laplacian_reverse.dif_LC_neumann

        if self._updates != self._mesh.updates() or self._l != l:
            self._updates = self._mesh.updates()
            self._v = self._mesh.get_vertices()
            self._dif_v = dif_v
            self._l = l

            self.dif_kappa_G = self._calc_dif_kappa_G()
            self.dif_vertex_N = self._calc_dif_vertex_N()
            self.dif_mean_curvature_normal = self._calc_dif_mean_curvature_normal()
            self.dif_kappa_H = self._calc_dif_kappa_H()
            self.dif_kappa_1 = self._calc_dif_kappa_1()
            self.dif_kappa_2 = self._calc_dif_kappa_2()
