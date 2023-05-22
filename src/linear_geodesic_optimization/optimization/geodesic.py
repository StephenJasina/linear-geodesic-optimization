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
        self.path_vertices: typing.List[dcelmesh.Mesh.Vertex] = []
        """Vertices through which the path passes."""
        self.path_halfedges: typing.List[typing.List[dcelmesh.Mesh.Halfedge]] \
            = []
        """
        A list of lists of halfedges through which the path passes.

        These halfedges are partitioned by `path_vertices`.
        """
        self.path_ratios: typing.List[typing.List[np.float64]] = []
        """
        Where along each halfedge the geodesic path passes through.

        Along with `path_halfedges` and `path_vertices`, this gives an
        easy way to reconstruct the actual path: simply linearly
        interpolate between the two endpoints of each halfedge using the
        corresponding ratio.
        """
        self.point_locations: typing.Dict[int, npt.NDArray[np.float64]] = {}
        """
        A map from vertex indices to their locations in 2-d.

        These points are found by "unfolding" the mesh along the
        geodesic path. The points are chosen so that the path is
        ultimately horizontal and starts at the origin.
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

    def _get_point_locations(self,
                             start: dcelmesh.Mesh.Vertex,
                             middle: typing.List[dcelmesh.Mesh.Halfedge],
                             end: dcelmesh.Mesh.Vertex,
                             initial_point: npt.NDArray[np.float64]) \
            -> typing.Dict[int, npt.NDArray[np.float64]]:
        """
        Unfold a sequence of faces according to the connecting edges.

        Return the locations of the vertices incident to the faces on
        the geodesic path when the mesh is "unfolded."

        Notably, this is a two-dimensional representation.
        """
        point_locations: typing.Dict[int, npt.NDArray[np.float64]] = {}
        """
        Map from vertex indices to point locations in two dimensions.
        """

        # Need to deal with the special case where the path is just a
        # single segment
        if not middle:
            point_locations[start.index()] = np.copy(initial_point)
            point_locations[end.index()] = np.array([
                np.linalg.norm(self._coordinates[start.index()]
                               - self._coordinates[end.index()]),
                0.
            ]) + initial_point
            return point_locations

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

        # Rotate and translate the points so that the geodesic path is
        # horizontal and starts at the initial point.
        end_start = point_locations[end.index()] \
            - point_locations[start.index()]
        end_start_norm = np.linalg.norm(end_start)
        c = end_start[0] / end_start_norm
        s = -end_start[1] / end_start_norm
        rotation = np.array([[c, -s], [s, c]])
        translation = initial_point - rotation @ point_locations[start.index()]
        for index in point_locations:
            point_locations[index] = rotation @ point_locations[index] \
                + translation

        return point_locations

    def _get_nearby_edges(self,
                          start: dcelmesh.Mesh.Vertex,
                          middle: typing.List[dcelmesh.Mesh.Halfedge],
                          end: dcelmesh.Mesh.Vertex) \
            -> typing.List[typing.Union[dcelmesh.Mesh.Halfedge,
                                        dcelmesh.Mesh.Edge]]:
        """
        Return a list of edges nearby the inputs.

        In particular, given endpoint and a sequence of edges that a
        path passes through intermediately, find the edges incident to
        the faces through which the path passes. For the edges on the
        boundary (i.e., the ones not part of `middle`), halfedges are
        returned.
        """
        middle_edges: typing.List[typing.Union[dcelmesh.Mesh.Halfedge,
                                               dcelmesh.Mesh.Edge]] = []
        """
        List of edges from near the center of the path.

        If an edge is in the "interior" (belongs to two of the faces),
        then it is stored as a `dcelmesh.Mesh.Edge`. Otherwise, it is
        stored as a `dcelmesh.Mesh.Halfedge`.
        """

        if not middle:
            return [self._topology.get_edge(start.index(), end.index())]

        for previous_halfedge, current_halfedge in itertools.pairwise(middle):
            if previous_halfedge.previous().origin().index() \
                    == current_halfedge.origin().index():
                middle_edges.append(previous_halfedge.previous())
            else:
                middle_edges.append(previous_halfedge.next())
            middle_edges.append(current_halfedge.edge())

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

        return start_edges + middle_edges + end_edges

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
        mu_path, mu_path_ratios = meshutility.pygeodesic.find_path(
            self._coordinates, self._faces, self._u, self._v
        )

        # Split the path up piecewise, where boundaries are marked by
        # vertices. Make sure to orient path_edges sensibly: each
        # halfedge points to the next face the path passes through.
        self.path_vertices = [self._topology.get_vertex(mu_path[0][0])]
        self.path_halfedges = []
        halfedges_to_add: typing.List[dcelmesh.Mesh.Halfedge] = []
        ratios_to_add: typing.List[np.float64] = []
        for index in range(1, len(mu_path)):
            i, j = mu_path[index]

            # Vertex case
            if i == j:
                self.path_vertices.append(self._topology.get_vertex(i))
                self.path_halfedges.append(halfedges_to_add)
                self.path_ratios.append(ratios_to_add)

                halfedges_to_add = []
                ratios_to_add = []
                continue

            # Pick the right direction for the halfedge
            halfedge_ij = self._topology.get_halfedge(i, j)
            if halfedge_ij.previous().origin().index() in mu_path[index + 1]:
                halfedges_to_add.append(halfedge_ij)
                ratios_to_add.append(mu_path_ratios[index])
            else:
                halfedges_to_add.append(self._topology.get_halfedge(j, i))
                ratios_to_add.append(1 - mu_path_ratios[index])

        # Compute point locations and the total geodesic distance
        self.point_locations = {}
        self.distance = np.float64(0.)
        current_point = np.zeros(2)
        for (start, end), middle in zip(itertools.pairwise(self.path_vertices),
                                        self.path_halfedges):
            point_locations \
                = self._get_point_locations(start, middle, end, current_point)
            for index, location in point_locations.items():
                self.point_locations[index] = location
            self.distance += np.linalg.norm(
                point_locations[end.index()]
                - point_locations[start.index()]
            )
            current_point = point_locations[end.index()]

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
        partials: typing.Dict[int, np.float64] = {}
        partials[start.index()] = np.float64(0.)
        partials[end.index()] = np.float64(0.)
        for halfedge in middle:
            partials[halfedge.origin().index()] = np.float64(0.)
            partials[halfedge.destination().index()] = np.float64(0.)
        """
        Partials with respect to vertices.

        These will be computed via accumulation.
        """

        geodesic = self.point_locations[end.index()] \
            - self.point_locations[start.index()]
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
            = self._topology.get_halfedge(middle[-1].destination().index(),
                                          end.index())

        for element in self._get_nearby_edges(start, middle, end):
            # Set the values:
            # * `edge`: The edge we're considering
            # * `partial_edge`: The partial of the geodesic length with
            #   respect to the length of the current edge
            # * `u`: One endpoint of the edge
            # * `v`: The other endpoint of the edge

            if isinstance(element, dcelmesh.Mesh.Halfedge):
                # Deal with the case where the edge is on the "boundary"

                # u and v are the endpoints of the halfedge. w is the
                # opposing vertex.
                u = element.origin()
                v = element.destination()
                w = element.previous().origin()
                edge = element.edge()

                print(f'{u.index()}, {v.index()}, exterior ({w.index()})')

                # Use bad names here and elsewhere to avoid lines
                # becoming too long and (even more) unreadable
                sw = self.point_locations[start.index()] \
                    - self.point_locations[w.index()]
                ew = self.point_locations[end.index()] \
                    - self.point_locations[w.index()]
                uw = self.point_locations[u.index()] \
                    - self.point_locations[w.index()]
                vw = self.point_locations[v.index()] \
                    - self.point_locations[w.index()]
                vu = self.point_locations[v.index()] \
                    - self.point_locations[u.index()]
                d_vu = np.linalg.norm(vu)
                partial_edge = d_vu * np.abs(np.cross(sw, ew)) \
                    / (d_geodesic * np.abs(np.cross(uw, vw)))

            else:
                # Deal with the case where the edge is in the "interior"

                # u and v will be the endpoints of the edge. w will be
                # the "previous" vertex (one step closer to the start),
                # and x will be the "next" vertex (one step closer to
                # the end).
                edge = element

                previous_halfedge = previous_halfedges[element.index()]
                next_halfedge = next_halfedges[element.index()]

                if previous_halfedge.origin().index() \
                        == next_halfedge.origin().index():
                    # Deal with the case where the previous and next
                    # edges share an endpoint
                    w = previous_halfedge.destination()
                    x = next_halfedge.destination()

                    # Set u to be the shared endpoint
                    potential = previous_halfedge.previous().origin()
                    edge_u, edge_v = edge.vertices()
                    if potential.index() == edge_u.index() \
                            or potential.index() == edge_v.index():
                        u = previous_halfedge.origin()
                        v = previous_halfedge.previous().origin()
                    else:
                        u = next_halfedge.origin()
                        v = next_halfedge.previous().origin()

                    print(f'{u.index()}, {v.index()}, interior shared ({w.index()}, {x.index()})')

                    su = self.point_locations[start.index()] \
                        - self.point_locations[u.index()]
                    wu = self.point_locations[w.index()] \
                        - self.point_locations[u.index()]
                    wv = self.point_locations[w.index()] \
                        - self.point_locations[v.index()]
                    eu = self.point_locations[end.index()] \
                        - self.point_locations[u.index()]
                    xu = self.point_locations[x.index()] \
                        - self.point_locations[u.index()]
                    xv = self.point_locations[x.index()] \
                        - self.point_locations[v.index()]
                    vu = self.point_locations[v.index()] \
                        - self.point_locations[u.index()]
                    d_vu = np.linalg.norm(vu)

                    partial_edge = -d_vu * np.abs(np.cross(su, eu)) \
                        * (1. / np.abs(np.cross(wu, wv))
                           + 1. / np.abs(np.cross(xu, xv))) \
                        / (d_geodesic
                           * (1. + np.abs(np.cross(wu, xu))
                              / np.abs(np.cross(wv, xv))))
                else:
                    # Deal with the case where the previous and next
                    # edges don't share an endpoint

                    # Set u to be closer to the start than v
                    u = previous_halfedge.origin()
                    v = next_halfedge.origin()
                    w = previous_halfedge.destination()
                    x = next_halfedge.destination()

                    print(f'{u.index()}, {v.index()}, interior unshared ({w.index()}, {x.index()})')

                    wu = self.point_locations[w.index()] \
                        - self.point_locations[u.index()]
                    wv = self.point_locations[w.index()] \
                        - self.point_locations[v.index()]
                    su = self.point_locations[start.index()] \
                        - self.point_locations[u.index()]
                    sv = self.point_locations[start.index()] \
                        - self.point_locations[v.index()]
                    sw = self.point_locations[start.index()] \
                        - self.point_locations[w.index()]
                    xu = self.point_locations[x.index()] \
                        - self.point_locations[u.index()]
                    xv = self.point_locations[x.index()] \
                        - self.point_locations[v.index()]
                    eu = self.point_locations[end.index()] \
                        - self.point_locations[u.index()]
                    ev = self.point_locations[end.index()] \
                        - self.point_locations[v.index()]
                    ex = self.point_locations[end.index()] \
                        - self.point_locations[x.index()]
                    vu = self.point_locations[v.index()] \
                        - self.point_locations[u.index()]
                    d_vu = np.linalg.norm(vu)

                    partial_edge = d_vu * (
                        (
                            np.abs(np.cross(sw, wv)) / np.abs(np.cross(wu, wv))
                            * (
                                (1. - sv @ ev / (sv @ sv)) / np.abs(np.cross(sv, ev))
                                + (1. - sv @ su / (sv @ sv)) / np.abs(np.cross(sv, su))
                            )
                        ) + (
                            np.abs(np.cross(ex, xu)) / np.abs(np.cross(xv, xu))
                            * (
                                (1. - eu @ su / (eu @ eu)) / np.abs(np.cross(eu, su))
                                + (1. - eu @ ev / (eu @ eu)) / np.abs(np.cross(eu, ev))
                            )
                        )
                        - 1. / np.abs(np.cross(sv, su))
                        - 1. / np.abs(np.cross(eu, ev))
                    ) / (d_geodesic * (
                        1. / np.abs(np.cross(sv, ev))
                        + 1. / np.abs(np.cross(eu, su))
                    ))

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

        # Set up for accumulation
        self.dif_distance = {
            index: np.float64(0.)
            for index in self.point_locations
        }

        # Finally, we actually do the accumulation
        for (start, end), middle in zip(itertools.pairwise(self.path_vertices),
                                        self.path_halfedges):
            for index, partial \
                    in self._reverse_part(start, middle, end).items():
                self.dif_distance[index] += partial
