"""Module containing utilities to compute geodesic paths."""

import itertools
import typing

import dcelmesh
import meshutility
import numpy as np
import numpy.typing as npt

from linear_geodesic_optimization.mesh.mesh import Mesh


class Computer:
    """
    Implementation of the fast marching on a mesh.

    This is essentially a wrapper around `meshutility`'s fast marching
    implementation with a reverse direction.
    """

    def __init__(self, mesh: Mesh, u: int, v: int):
        """
        Initialize the computer.

        As input, the computer accepts a mesh and the indices of two
        special vertices for which the geodesic path will be computed.
        """
        self._mesh: Mesh = mesh
        self._topology: dcelmesh.Mesh = mesh.get_topology()
        self._faces: typing.List[typing.Tuple[int, ...]] = [
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
        self.edge_lengths: typing.List[np.float64] \
            = [np.float64(0.) for _ in self._topology.edges()]
        """A list of the mesh's edge lengths, indexed by edges."""
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
        self.distance: np.float64 = np.float64(0.)
        """The geodesic distance itself."""

        # Reverse variables
        self._reverse_updates: int = self._mesh.get_updates() - 1
        self._partials: npt.NDArray[np.float64] \
            = np.zeros((self._topology.n_vertices(), 3))
        self.dif_edge_lengths: typing.List[typing.Dict[int, np.float64]] \
            = [{} for _ in self._topology.edges()]
        """
        A list of the partials of the mesh's edge lengths, indexed by
        edges, and then by vertices.
        """
        self.dif_distance: typing.Dict[int, np.float64] = {}
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

        # Compute edge lengths
        for edge in self._topology.edges():
            u, v = edge.vertices()
            self.edge_lengths[edge.index()] \
                = np.linalg.norm(self._coordinates[u.index()]
                                 - self._coordinates[v.index()])

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

            # Vertex case
            if i == j:
                self.path.append(self._topology.get_vertex(i))
                self.path_ratios.append(np.float64(0.))
                continue

            # Pick the right direction for the halfedge
            halfedge_ij = self._topology.get_halfedge(i, j)
            if halfedge_ij.previous().origin().index() in path[index + 1]:
                self.path.append(halfedge_ij)
                self.path_ratios.append(path_ratios[index])
            else:
                self.path.append(self._topology.get_halfedge(j, i))
                self.path_ratios.append(1 - path_ratios[index])

        # Reconstruct the path. This follows the example code from
        # https://zishun.github.io/projects/MeshUtility/
        points_0 = self._coordinates[path[:, 0]]
        points_1 = self._coordinates[path[:, 1]]
        points: npt.NDArray[np.float64] \
            = np.multiply(points_0, 1. - path_ratios[:, np.newaxis]) \
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
        d_w = np.linalg.norm(v - u)
        h = (d_w**2 + d_v**2 - d_u**2) / (2. * d_w)
        k = np.sqrt(d_v**2 - h**2)
        rotate = np.array([[0., -1.], [1., 0.]])
        direction = (v - u) / d_w
        return u + h * direction + k * (rotate @ direction)

    def _calc_point_locations(self,
                              start: dcelmesh.Mesh.Vertex,
                              middle: typing.List[dcelmesh.Mesh.Halfedge],
                              end: dcelmesh.Mesh.Vertex) \
            -> typing.Tuple[typing.Dict[int, npt.NDArray[np.float64]],
                            typing.List[typing.Union[dcelmesh.Mesh.Halfedge,
                                                     dcelmesh.Mesh.Edge]]]:
        """
        Unfold a sequence of faces according to the connecting edges.

        Return the locations of the vertices incident to the faces on
        the geodesic path when the mesh is "unfolded," as well as a list
        of the edges used (halfedges on the boundary).

        Notably, this is a two-dimensional representation.
        """
        point_locations: typing.Dict[int, npt.NDArray[np.float64]] = {}
        """
        Map from vertex indices to point locations in two dimensions.
        """
        middle_edges: typing.List[typing.Union[dcelmesh.Mesh.Halfedge,
                                               dcelmesh.Mesh.Edge]] = []
        """
        List of edges from near the center of the path.

        If an edge is in the "interior" (belongs to two of the faces),
        then it is stored as a `dcelmesh.Mesh.Edge`. Otherwise, it is
        stored as a `dcelmesh.Mesh.Halfedge`.
        """

        # Need to deal with the special case where the path is just a
        # single segment
        if not middle:
            point_locations[start.index()] = np.zeros(2)
            point_locations[end.index()] = np.array([
                np.linalg.norm(self._coordinates[start.index()]
                               - self._coordinates[end.index()]),
                0.
            ])
            return point_locations, [self._topology.get_edge(start.index(),
                                                             end.index())]

        # Start by placing the first edge from the middle
        point_locations[middle[0].origin().index()] = np.zeros(2)
        point_locations[middle[0].destination().index()] = np.array([
            np.linalg.norm(
                self._coordinates[middle[0].origin().index()]
                - self._coordinates[middle[0].destination().index()]),
            0.
        ])

        # Place the remaining edges from the middle
        for previous_halfedge, current_halfedge in itertools.pairwise(middle):
            i = previous_halfedge.origin().index()
            j = previous_halfedge.destination().index()
            k = previous_halfedge.previous().origin().index()

            # v_i and v_j have already had their locations computed. We
            # need to make sure v_k is the right point to add.
            if k != current_halfedge.origin().index() \
                    and k != current_halfedge.destination().index():
                i, j = j, i
                twin = previous_halfedge.twin()
                if twin is None:
                    raise dcelmesh.Mesh.IllegalMeshException(
                        f'Halfedge ({previous_halfedge.origin().index()}, '
                        f'{previous_halfedge.destination().index()}) '
                        'has no twin'
                    )
                k = twin.previous().origin().index()

            point_locations[k] = Computer._get_next_point(
                point_locations[i],
                point_locations[j],
                np.linalg.norm(self._coordinates[k] - self._coordinates[j]),
                np.linalg.norm(self._coordinates[k] - self._coordinates[i])
            )

            # Upkeep the list of edges
            if k == current_halfedge.origin().index():
                middle_edges.append(previous_halfedge.previous())
            else:
                middle_edges.append(previous_halfedge.next())
            middle_edges.append(current_halfedge.edge())

        # Place the starting point
        i = middle[0].destination().index()
        j = middle[0].origin().index()
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
        k = end.index()
        point_locations[k] = Computer._get_next_point(
            point_locations[i],
            point_locations[j],
            np.linalg.norm(self._coordinates[k] - self._coordinates[j]),
            np.linalg.norm(self._coordinates[k] - self._coordinates[i])
        )

        # Add the remaining edges
        twin = middle[0].twin()
        if twin is None:
            raise dcelmesh.Mesh.IllegalMeshException(
                f'Halfedge ({middle[0].origin().index()}, '
                f'{middle[0].origin().index()}) has no twin'
            )
        start_edges: typing.List[typing.Union[dcelmesh.Mesh.Halfedge,
                                              dcelmesh.Mesh.Edge]] = [
            twin.next(),
            twin.previous(),
            middle[0].edge()
        ]
        end_edges: typing.List[typing.Union[dcelmesh.Mesh.Halfedge,
                                            dcelmesh.Mesh.Edge]] = [
            middle[-1].next(),
            middle[-1].previous()
        ]

        return point_locations, start_edges + middle_edges + end_edges

    def _reverse_part(self,
                      start: dcelmesh.Mesh.Vertex,
                      middle: typing.List[dcelmesh.Mesh.Halfedge],
                      end: dcelmesh.Mesh.Vertex) \
            -> typing.Dict[int, np.float64]:
        """
        Compute the partials for a geodesic not passing through saddles.

        In other words, the only mesh points the geodesic path should
        coincide with are the endpoints.
        """
        point_locations, edges = self._calc_point_locations(start, middle, end)
        partials: typing.Dict[int, np.float64] = {
            index: np.float64(0.)
            for index in point_locations
        }
        """
        Partials with respect to vertices.

        These will be computed via accumulation.
        """

        geodesic = point_locations[end.index()] \
            - point_locations[start.index()]
        d_geodesic = np.linalg.norm(geodesic)

        # Deal with the case where there are no faces
        if not middle:
            edge = self._topology.get_edge(start.index(), end.index())
            partials[start.index()] \
                = self.dif_edge_lengths[edge.index()][start.index()]
            partials[end.index()] \
                = self.dif_edge_lengths[edge.index()][end.index()]
            return partials

        # Compute maps telling us the previous and next edges the path
        # passes through. The halfedges are oriented so that their
        # origins lie on the edge in question.
        previous_halfedges: typing.Dict[int, dcelmesh.Mesh.Halfedge] = {}
        for previous, current in itertools.pairwise(middle):
            if (previous.origin().index() == current.origin().index()
                    or previous.origin().index()
                    == current.destination().index()):
                previous_halfedges[current.edge().index()] = previous
            else:
                twin = previous.twin()
                if twin is None:
                    raise dcelmesh.Mesh.IllegalMeshException(
                        f'Halfedge ({previous.origin().index()}, '
                        f'{previous.destination().index()}) has no twin'
                    )
                previous_halfedges[current.edge().index()] = twin
        previous_halfedges[middle[0].edge().index()] \
            = self._topology.get_halfedge(middle[0].origin().index(),
                                          start.index())
        next_halfedges: typing.Dict[int, dcelmesh.Mesh.Halfedge] = {}
        for previous, current in itertools.pairwise(middle):
            if (current.origin().index() == previous.origin().index()
                    or current.origin().index()
                    == previous.destination().index()):
                next_halfedges[previous.edge().index()] = current
            else:
                twin = current.twin()
                if twin is None:
                    raise dcelmesh.Mesh.IllegalMeshException(
                        f'Halfedge ({current.origin().index()}, '
                        f'{current.destination().index()}) has no twin'
                    )
                next_halfedges[previous.edge().index()] = twin
        next_halfedges[middle[-1].edge().index()] \
            = self._topology.get_halfedge(middle[-1].origin().index(),
                                          end.index())

        for element in edges:
            # Set the values:
            # * `edge`: The edge we're considering
            # * `partial_edge`: The partial of the geodesic length with
            #   respect to the length of the current edge
            # * `u`: One endpoint of the edge
            # * `v`: The other endpoint of the edge

            # Deal with the case where the edge is on the "boundary"
            if isinstance(element, dcelmesh.Mesh.Halfedge):
                # u and v are the endpoints of the halfedge. w is the
                # opposing vertex.
                u = element.origin()
                v = element.destination()
                w = element.previous().origin()
                edge = element.edge()

                start_w = point_locations[start.index()] \
                    - point_locations[w.index()]
                end_w = point_locations[end.index()] \
                    - point_locations[w.index()]
                u_w = point_locations[u.index()] \
                    - point_locations[w.index()]
                v_w = point_locations[v.index()] \
                    - point_locations[w.index()]
                v_u = point_locations[v.index()] \
                    - point_locations[u.index()]
                d_v_u = np.linalg.norm(v_u)

                partial_edge = d_v_u * abs(np.cross(start_w, end_w)) \
                    / (d_geodesic * abs(np.cross(u_w, v_w)))

            else:
                edge = element

                previous_halfedge = previous_halfedges[element.index()]
                next_halfedge = next_halfedges[element.index()]

                # Deal with the case where the previous and next edges
                # share and endpoint
                if previous_halfedge.origin().index() \
                        == next_halfedge.origin().index():
                    prev = previous_halfedge.destination()
                    nxt = next_halfedge.destination()
                    potential = previous_halfedge.previous().origin()
                    edge_u, edge_v = edge.vertices()
                    if potential.index() == edge_u.index() \
                            or potential.index() == edge_v.index():
                        u = previous_halfedge.origin()
                        v = previous_halfedge.previous().origin()
                    else:
                        u = next_halfedge.origin()
                        v = next_halfedge.previous().origin()

                    start_u = point_locations[start.index()] \
                        - point_locations[u.index()]
                    prev_u = point_locations[prev.index()] \
                        - point_locations[u.index()]
                    prev_v = point_locations[prev.index()] \
                        - point_locations[v.index()]
                    end_u = point_locations[end.index()] \
                        - point_locations[u.index()]
                    nxt_u = point_locations[nxt.index()] \
                        - point_locations[u.index()]
                    nxt_v = point_locations[nxt.index()] \
                        - point_locations[v.index()]
                    v_u = point_locations[v.index()] \
                        - point_locations[u.index()]
                    d_v_u = np.linalg.norm(v_u)

                    partial_edge = -d_v_u \
                        * np.cross(start_u, end_u) \
                        * (1 / np.cross(prev_u, prev_v)
                           + 1 / np.cross(nxt_u, nxt_v)) \
                        / (d_geodesic * (1 + np.cross(prev_u, nxt_u)
                                         / np.cross(prev_v, nxt_v)))
                else:
                    u = previous_halfedge.origin()
                    v = next_halfedge.origin()
                    partial_edge = np.float64(0.)

            partials[u.index()] += partial_edge \
                * self.dif_edge_lengths[edge.index()][u.index()]
            partials[v.index()] += partial_edge \
                * self.dif_edge_lengths[edge.index()][v.index()]

        return partials

    def reverse(self) -> None:
        """
        Compute the reverse direction (that is, partials).

        The computed values will be stored in the variable
        `Computer.dif_distance`.
        """
        self.forward()
        if self._reverse_updates == self._mesh.get_updates():
            return
        self._reverse_updates = self._mesh.get_updates()
        self._partials = self._mesh.get_partials()

        # Compute the partials of edge lengths first
        for edge in self._topology.edges():
            u, v = edge.vertices()
            pu = self._coordinates[u.index()]
            pv = self._coordinates[v.index()]
            edge_length = self.edge_lengths[edge.index()]
            self.dif_edge_lengths[edge.index()][u.index()] \
                = (pu - pv) @ self._partials[u.index()] / edge_length
            self.dif_edge_lengths[edge.index()][v.index()] \
                = (pv - pu) @ self._partials[v.index()] / edge_length

        # Split the path up piecewise, where boundaries are marked by
        # vertices.
        # At the same time, dif_distance will be computed by
        # accumulation, so we set the relevant values to zero.
        path_vertices: typing.List[dcelmesh.Mesh.Vertex] = []
        path_halfedges: typing.List[typing.List[dcelmesh.Mesh.Halfedge]] = []
        self.dif_distance = {}
        for element in self.path:
            if isinstance(element, dcelmesh.Mesh.Vertex):
                path_vertices.append(element)
                path_halfedges.append([])
                self.dif_distance[element.index()] = np.float64(0.)
            else:
                path_halfedges[-1].append(element)
                self.dif_distance[element.origin().index()] = np.float64(0.)
                self.dif_distance[element.destination().index()] \
                    = np.float64(0.)
        # Remove the extraneous empty list
        del path_halfedges[-1]

        # Finally, we actually do the accumulation
        for (start, end), middle in zip(itertools.pairwise(path_vertices),
                                        path_halfedges):
            for key, value in self._reverse_part(start, middle, end).items():
                self.dif_distance[key] += value
