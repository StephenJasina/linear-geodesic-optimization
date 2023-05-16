"""Module containing utilities to compute geodesic paths."""

import collections
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

        self._u: int = u
        self._v: int = v

        # Forward variables
        self._forward_updates: int = self._mesh.get_updates() - 1
        self._coordinates: npt.NDArray[np.float64] \
            = np.zeros((self._topology.n_vertices(), 3))
        self.path_edges: typing.List[typing.Tuple[int, int]] = []
        """
        A list of edges that the geodesic path passes through.

        For endpoints and saddle points through which path passes, the
        pairs consist of two copies of the same point.
        """
        self.path_ratios: typing.List[float] = []
        """
        Where along each edge the geodesic path passes through.

        Along with `path_edges`, this gives an easy way to reconstruct
        the actual path: simply linearly interpolate between the two
        endpoints of each edge using the corresponding ratio.
        """
        self.distance: float = 0.
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

    def forward(self):
        """
        Compute the forward direction.

        The computed values will be stored in the variables:
        * `Computer.path_edges`
        * `Computer.path_ratios`
        * `Computer.distance`
        """
        if self._forward_updates == self._mesh.get_updates():
            return
        self._forward_updates = self._mesh.get_updates()
        self._coordinates = self._mesh.get_coordinates()

        # Call the meshutility solver
        self.path_edges, self.path_ratios = meshutility.pygeodesic.find_path(
            self._coordinates, self._faces, self._u, self._v
        )

        # Reconstruct the path
        points0 = self._coordinates[self.path_edges[:,0]]
        points1 = self._coordinates[self.path_edges[:,1]]
        points = np.multiply(points0, 1. - self.path_ratios[:, np.newaxis]) \
            + np.multiply(points1, self.path_ratios[:, np.newaxis])

        # The total distance is the sum of the length of each segment
        self.distance = sum([
            np.linalg.norm(a - b)
            for a, b in itertools.pairwise(points)
        ])

    # @staticmethod
    # def next_():
    #     pass

    # def _calc_single_edge(self, a, b):
    #     return 0.

    # def _calc_convex(self, path_edges, path_ratios):
    #     point_locations = {}
    #     point_locations[path_edges[0][0]] = self._vertices[path_edges[0][0]]
    #     point_locations[path_edges[1][0]] = self._vertices[path_edges[1][0]]
    #     point_locations[path_edges[1][1]] = self._vertices[path_edges[1][1]]
    #     for i, (u, v) in enumerate(path_edges[2:], start=2):
    #         if self._nxt[u,v] in path_edges[i-1]:
    #             u, v = v, u
    #         t = self._nxt[v,u]
    #         w = self._nxt[u,v]

    #         # Want w to be coplanar to t, u, and v
    #         point_locations[w] = 0. # TODO

    #     partials = collections.defaultdict(float)
    #     mesh_partials = self._mesh.get_partials()

    #     return partials

    def reverse(self):
        self.forward()
        self._laplacian.reverse()
        if self._reverse_updates == self._mesh.get_updates():
            return
        self._reverse_updates = self._mesh.get_updates()

        saddle_indices = [i for i, (a, b) in enumerate(self.path_edges) if a == b]
        print(saddle_indices)

        self.partials = collections.defaultdict(float)
        for i, j in itertools.pairwise(saddle_indices):
            for key, value in self._calc_convex(self._path_edges[i:j+1], self._path_ratios[i:j+1]):
                self.partials[key] += value
