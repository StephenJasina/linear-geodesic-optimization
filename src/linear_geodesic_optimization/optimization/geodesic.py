"""Module containing utilities to compute geodesic paths."""

import itertools
import typing

import dcelmesh
import meshutility
import numpy as np
import numpy.typing as npt

from linear_geodesic_optimization.mesh.mesh import Mesh
from linear_geodesic_optimization.optimization.laplacian \
    import Computer as Laplacian


class Computer:
    """
    Implementation of the fast marching on a mesh.

    This is essentially a wrapper around `meshutility`'s fast marching
    implementation with a reverse direction.
    """

    def __init__(self, mesh: Mesh, u: int, v: int, laplacian: Laplacian):
        """
        Initialize the computer.

        As input, the computer accepts a mesh and the indices of two
        special vertices for which the geodesic path will be computed. A
        `laplacian.Computer` also must be passed in for the reverse
        computations.
        """
        self._mesh: Mesh = mesh
        self._topology: dcelmesh.Mesh = mesh.get_topology()
        self._faces: typing.List[typing.Tuple[int, int, int]] = [
            tuple(vertex.index() for vertex in face.vertices())
            for face in self._topology.faces()
        ]
        """
        An explicit representation of our mesh's topology. This is
        required as input to `meshutility`'s geodesic solver.
        """

        self._u: int = u
        self._v: int = v

        # Forward variables
        self._forward_updates: int = self._mesh.get_updates() - 1
        self._coordinates: npt.NDArray[np.float64] \
            = np.zeros((self._topology.n_vertices(), 3))
        self.path: typing.List[typing.Union[dcelmesh.Mesh.Vertex,
                                            dcelmesh.Mesh.Halfedge]] = []
        """
        A list of vertices and edges incident to the geodesic path.

        For endpoints and saddle points through which path passes, the
        corresponding element is just a vertex. For other edges, the
        elements are halfedges oriented such that the face the halfedge
        points towards contains the next vertex or halfedge (i.e., its
        the next face the path passes through).
        """
        self.path_ratios: typing.List[np.float64] = []
        """
        Where along each edge the geodesic path passes through.

        Along with `path_edges`, this gives an easy way to reconstruct
        the actual path: simply linearly interpolate between the two
        endpoints of each edge using the corresponding ratio.
        """
        self.distance: np.float64 = 0.
        """The geodesic distance itself."""

        # Reverse variables
        self._reverse_updates: int = self._mesh.get_updates() - 1
        self._partials: npt.NDArray[np.float64] \
            = np.zeros((self._topology.n_vertices(), 3))
        self._laplacian: Laplacian = laplacian
        self.dif_distance: typing.Dict[int, float] = {}
        """
        The partials of the geodesic distance, indexed by vertex.

        Note that the only vertices for which this dictionary is
        populated are those that are incident to the faces through which
        the geodesic path passes.
        """

    def forward(self) -> None:
        """
        Compute the forward direction.

        The computed values will be stored in the variables:
        * `Computer.path`
        * `Computer.path_ratios`
        * `Computer.distance`
        """
        if self._forward_updates == self._mesh.get_updates():
            return
        self._forward_updates = self._mesh.get_updates()
        self._coordinates = self._mesh.get_coordinates()

        # Call the meshutility solver
        path, path_ratios = meshutility.pygeodesic.find_path(
            self._coordinates, self._faces, self._u, self._v
        )

        # Orient path_edges sensibly. Each halfedge points to the next
        # face the path passes through.
        self.path = []
        self.path_ratios = []
        for index in range(len(path)):
            i, j = path[index]
            if i == j:
                self.path.append(self._topology.get_vertex(i))
                self.path_ratios.append(np.float64(0.))
            elif self._topology.get_halfedge(i, j).previous().origin() \
                    in path[index + 1]:
                self.path.append(self._topology.get_halfedge(i, j))
                self.path_ratios.append(path_ratios[index])
            else:
                self.path.append(self._topology.get_halfedge(j, i))
                self.path_ratios.append(1 - path_ratios[index])

        # Reconstruct the path. This follows the example code from
        # https://zishun.github.io/projects/MeshUtility/
        points_0 = self._coordinates[path[:, 0]]
        points_1 = self._coordinates[path[:, 1]]
        points = np.multiply(points_0, 1. - path_ratios[:, np.newaxis]) \
            + np.multiply(points_1, path_ratios[:, np.newaxis])

        # The total distance is the sum of the length of each segment
        self.distance = sum([
            np.linalg.norm(a - b)
            for a, b in itertools.pairwise(points)
        ])

    @staticmethod
    def _get_next_point(
        u: npt.NDArray,
        v: npt.NDArray,
        d_u: np.float64,
        d_v: np.float64
    ) -> npt.NDArray:
        """
        Find a point at a certain distance from two other points.

        Given two two-dimensional input points `u` and `v`, find a point
        `w` so that the distance from `v` to `w` is `d_u` and the
        distance from `u` to `w` is `d_v`. Furthermore, ensure that the
        resulting triangle (`u`, `v`, `w`) is oriented counterclockwise.
        """
        d_w = np.linalg.norm(u - v)
        h = (d_w + (d_v**2 - d_u**2) / d_w) / 2.
        k = np.sqrt(d_v**2 - h**2)
        rotate = np.array([[0., -1.], [1., 0.]])
        return u + h * (v - u) + k * (rotate @ (v - u))

    def _calc_point_locations(self,
                              start: dcelmesh.Mesh.Vertex,
                              middle: typing.List[dcelmesh.Mesh.Halfedge],
                              end: dcelmesh.Mesh.Vertex) \
            -> typing.Dict[int, npt.NDArray[np.float64]]:
        """
        Unfold a sequence of faces according to the connecting edges.

        Return the locations of the vertices incident to the faces on
        the geodesic path when the mesh is "unfolded." Notably, this is
        a two-dimensional representation.
        """
        point_locations: typing.Dict[int, npt.NDArray[np.float64]] = {}
        if not middle:
            point_locations[start.index()] = np.zeros(2)
            point_locations[end.index()] = np.array([
                np.linalg.norm(self._coordinates[start.index()]
                               - self._coordinates[end.index()]),
                0.
            ])
            return point_locations

        # Start by placing the first edge
        point_locations[middle[0].origin().index()] = np.zeros(2)
        point_locations[middle[0].destination().index()] = np.array([
            np.linalg.norm(
                self._coordinates[middle[0].origin().index()]
                - self._coordinates[middle[0].destination().index()]),
            0.
        ])

        # Place the remaining edges
        for previous_halfedge, current_halfedge in itertools.pairwise(middle):
            i = previous_halfedge.origin().index()
            j = previous_halfedge.destination().index()
            k = previous_halfedge.previous().origin().index()

            # v_i and v_j have already had their locations computed. We
            # need to make sure v_k is the right point to add.
            if k != current_halfedge.origin().index() \
                    and k != current_halfedge.destination().index():
                i, j = j, i
                k = previous_halfedge.twin().previous().origin().index()

            point_locations[k] = Computer._get_next_point(
                point_locations[i],
                point_locations[j],
                np.linalg.norm(self._coordinates[k] - self._coordinates[j]),
                np.linalg.norm(self._coordinates[k] - self._coordinates[i])
            )

        # Place the starting point
        i = middle[0].origin().index()
        j = middle[0].destination().index()
        if middle[0].previous().origin().index() != start.index():
            i, j = j, i
        k = start.index()
        point_locations[k] = Computer._get_next_point(
            point_locations[i],
            point_locations[j],
            np.linalg.norm(self._coordinates[k] - self._coordinates[j]),
            np.linalg.norm(self._coordinates[k] - self._coordinates[i])
        )

        # Place the ending point
        i = middle[-1].origin().index()
        j = middle[-1].destination().index()
        if middle[-1].previous().origin().index() != end.index():
            i, j = j, i
        k = end.index()
        point_locations[k] = Computer._get_next_point(
            point_locations[i],
            point_locations[j],
            np.linalg.norm(self._coordinates[k] - self._coordinates[j]),
            np.linalg.norm(self._coordinates[k] - self._coordinates[i])
        )

    def _calc_convex(self,
                     start: dcelmesh.Mesh.Vertex,
                     middle: typing.List[dcelmesh.Mesh.Halfedge],
                     end: dcelmesh.Mesh.Vertex) \
            -> typing.Dict[int, np.float64]:
        """
        Compute the partials for a geodesic not passing through saddles.

        In other words, the only mesh points the geodesic path should
        coincide with are the endpoints.
        """
        partials = {}
        mesh_partials = self._mesh.get_partials()

        return partials

    def reverse(self) -> None:
        self.forward()
        self._laplacian.reverse()
        if self._reverse_updates == self._mesh.get_updates():
            return
        self._reverse_updates = self._mesh.get_updates()
        self._partials = self._mesh.get_partials()

        # Split the path up piecewise, where boundaries are marked by
        # vertices.
        # At the same time, dif_distance will be computed by
        # accumulation, so we set the relevant values to zero.
        path_vertices: typing.List[dcelmesh.Mesh.Vertex] = []
        path_edges: typing.List[typing.List[dcelmesh.Mesh.Halfedge]] = []
        self.dif_distance = {}
        for element in self.path:
            if isinstance(element, dcelmesh.Mesh.Vertex):
                path_vertices.append(element)
                path_edges.append([])
                self.dif_distance[element.index()] = np.float64(0.)
            else:
                path_edges[-1].append(element)
                self.dif_distance[element.origin().index()] = np.float64(0.)
                self.dif_distance[element.destination().index()] \
                    = np.float64(0.)

        for (start, end), middle in zip(itertools.pairwise(path_vertices),
                                        path_edges):
            for key, value in self._calc_convex(start, middle, end):
                self.dif_distance[key] += value
