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
        self._start_points: typing.List[npt.NDArray[np.float64]] = []
        """List of the start locations for each path component."""
        self._end_points: typing.List[npt.NDArray[np.float64]] = []
        """List of the end locations for each path component."""
        self._double_boundary: typing.List[typing.List[typing.Tuple[
            dcelmesh.Mesh.Halfedge,
            npt.NDArray[np.float64],
            npt.NDArray[np.float64]
        ]]] = []
        """
        A list of lists of halfedges that appear on between two saddle
        vertices. Alternatively, one can think of this as halfedges on
        the boundary whose twins are also on the boundary. Additionally
        stored are the positions of the halfedges' origins and
        destinations.
        """
        self._boundary: typing.List[typing.List[typing.Tuple[
            dcelmesh.Mesh.Halfedge,
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64]
        ]]] = []
        """
        A list of lists of halfedges that appear on the boundary of the
        sequence of faces, as well as the positions of the vertices of
        the adjacent face. If a halfedge is (u, v) and the next halfedge
        is (v, w), then the positions are returned in the order
        (u, v, w).
        """
        self._interior_shared: typing.List[typing.List[typing.Tuple[
            dcelmesh.Mesh.Halfedge,
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64]
        ]]] = []
        """
        A list of lists of halfedges that appear between faces such that
        the previous and next connections are on adjacent sides of the
        quadrilateral formed by the two faces. Also included are
        positions of the four vertices making up the two faces in the
        order (u, v, w, x), where u is the vertex on the halfedge shared
        by the two connections, v is the vertex on the halfedge unshared
        by the two connections, w is the other vertex closer to the
        start, and x is other vertex closer to the end.
        """
        self._interior_unshared: typing.List[typing.List[typing.Tuple[
            dcelmesh.Mesh.Halfedge,
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64]
        ]]] = []
        """
        A list of lists of halfedges that appear between faces such that
        the previous and next connections are on opposing sides of the
        quadrilateral formed by the two faces. Also included are
        positions of the four vertices making up the two faces in the
        order (u, v, w, x), where u is the vertex on the halfedge closer
        to the start, v is the vertex on the halfedge closer to the end,
        w is the other vertex closer to the start, and x is the other
        vertex closer to the end.
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
        u: npt.NDArray[np.float64],
        v: npt.NDArray[np.float64],
        d_u: np.float64,
        d_v: np.float64
    ) -> npt.NDArray[np.float64]:
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
        rotate = np.array([[0., -1.], [1., 0.]], dtype=np.float64)
        direction = (v - u) / d_w
        return u + h * direction + k * (rotate @ direction)

    def _get_point_locations(self,
                             start: dcelmesh.Mesh.Vertex,
                             middle: typing.List[dcelmesh.Mesh.Halfedge],
                             end: dcelmesh.Mesh.Vertex) \
            -> typing.Tuple[
                npt.NDArray[np.float64],
                typing.List[typing.Tuple[
                    dcelmesh.Mesh.Halfedge,
                    npt.NDArray[np.float64],
                    npt.NDArray[np.float64]
                ]],
                typing.List[typing.Tuple[
                    dcelmesh.Mesh.Halfedge,
                    npt.NDArray[np.float64],
                    npt.NDArray[np.float64],
                    npt.NDArray[np.float64]
                ]],
                typing.List[typing.Tuple[
                    dcelmesh.Mesh.Halfedge,
                    npt.NDArray[np.float64],
                    npt.NDArray[np.float64],
                    npt.NDArray[np.float64],
                    npt.NDArray[np.float64]
                ]],
                typing.List[typing.Tuple[
                    dcelmesh.Mesh.Halfedge,
                    npt.NDArray[np.float64],
                    npt.NDArray[np.float64],
                    npt.NDArray[np.float64],
                    npt.NDArray[np.float64]
                ]],
                npt.NDArray[np.float64]
            ]:
        """
        Unfold a sequence of faces according to the connecting edges.

        Return six things:
        * The index and location of the starting vertex.
        * An appropriate value for an element of `self._boundary`
        * An appropriate value for an element of `self._double_boundary`
        * An appropriate value for an element of `self._interior_shared`
        * An appropriate value for an element of
          `self._interior_unshared`
        * The index and location of the ending vertex.

        Notably, this is a two-dimensional representation.
        """
        coordinates = self._coordinates

        double_boundary: typing.List[typing.Tuple[
            dcelmesh.Mesh.Halfedge,
            npt.NDArray[np.float64],
            npt.NDArray[np.float64]
        ]] = []
        boundary: typing.List[typing.Tuple[
            dcelmesh.Mesh.Halfedge,
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64]
        ]] = []
        interior_shared: typing.List[typing.Tuple[
            dcelmesh.Mesh.Halfedge,
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64]
        ]] = []
        interior_unshared: typing.List[typing.Tuple[
            dcelmesh.Mesh.Halfedge,
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64]
        ]] = []

        ps = np.zeros(2)

        # Need to deal with the special case where the path is just a
        # single segment
        if not middle:
            pe = np.array([
                np.linalg.norm(self._coordinates[start.index()]
                               - self._coordinates[end.index()]),
                0.
            ], dtype=np.float64)
            halfedge = next(self._topology.get_edge(start.index(),
                                                    end.index()).halfedges())
            if halfedge.origin().index() == start.index():
                return ps, [(halfedge, ps, pe)], [], [], [], pe
            else:
                return ps, [(halfedge, pe, ps)], [], [], [], pe

        previous_was_u_side = True
        u = middle[0].origin()
        v = middle[0].destination()
        w = start
        x = middle[0].previous().origin()
        pw = ps
        pu = np.array([
            np.linalg.norm(coordinates[u.index()]
                           - coordinates[start.index()]),
            np.float64(0.)
        ], dtype=np.float64)
        pv = self._get_next_point(
            pu,
            pw,
            np.linalg.norm(coordinates[v.index()] - coordinates[w.index()]),
            np.linalg.norm(coordinates[v.index()] - coordinates[u.index()])
        )
        px = self._get_next_point(
            pu,
            pv,
            np.linalg.norm(coordinates[x.index()] - coordinates[v.index()]),
            np.linalg.norm(coordinates[x.index()] - coordinates[u.index()])
        )

        # Place the first two boundary edges
        twin = middle[0].twin()
        if twin is None:
            raise dcelmesh.Mesh.IllegalMeshException(
                f'Halfedge ({middle[0].origin().index()}, '
                f'{middle[0].destination().index()}) '
                'has no twin'
            )
        boundary.append((twin.next(), pu, pw, pv))
        boundary.append((twin.previous(), pw, pv, pu))

        # Place the other boundary edges (except one) and the interior
        # edges
        for index in range(len(middle)):
            halfedge = middle[index]
            is_u_side = index == len(middle) - 1 \
                or u.index() == middle[index + 1].origin().index()

            if is_u_side:
                boundary.append((halfedge.next(), pv, px, pu))
                if previous_was_u_side:
                    interior_shared.append((halfedge, pu, pv, pw, px))
                else:
                    interior_unshared.append((halfedge, pv, pu, pw, px))

                w = v
                pw = pv
                v = x
                pv = px
            else:
                boundary.append((halfedge.previous(), px, pu, pv))
                if previous_was_u_side:
                    interior_unshared.append((halfedge, pu, pv, pw, px))
                else:
                    interior_shared.append((halfedge, pv, pu, pw, px))

                w = u
                pw = pu
                u = x
                pu = px

            if index == len(middle) - 1:
                break

            x = middle[index + 1].previous().origin()
            px = self._get_next_point(
                pu,
                pv,
                np.linalg.norm(coordinates[x.index()]
                               - coordinates[v.index()]),
                np.linalg.norm(coordinates[x.index()] - coordinates[u.index()])
            )

            previous_was_u_side = is_u_side

        # Place the last boundary edge
        boundary.append((middle[-1].previous(), pv, pu, pw))

        pe = px

        return ps, double_boundary, boundary, \
            interior_shared, interior_unshared, pe

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
        previous_vertex_index: typing.Optional[int] = mu_path[0][0]
        self.path_vertices = [self._topology.get_vertex(previous_vertex_index)]
        self.path_halfedges = []
        self.path_ratios = []
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

                previous_vertex_index = i
                continue

            # Guard against some strange floating point quirks that
            # the meshutility solver has
            if previous_vertex_index is not None \
                    and (previous_vertex_index == i
                         or previous_vertex_index == j):
                continue
            previous_vertex_index = None

            # Pick the right direction for the halfedge
            halfedge_ij = self._topology.get_halfedge(i, j)
            # This check is legal because, if index == len(mu_path) - 1,
            # then we fall into the vertex case.
            if halfedge_ij.previous().origin().index() in mu_path[index + 1]:
                halfedges_to_add.append(halfedge_ij)
                ratios_to_add.append(mu_path_ratios[index])
            else:
                halfedges_to_add.append(self._topology.get_halfedge(j, i))
                ratios_to_add.append(1 - mu_path_ratios[index])

        # Compute point locations and the total geodesic distance
        self._start_points = []
        self._end_points = []
        self._double_boundary = []
        self._boundary = []
        self._interior_shared = []
        self._interior_unshared = []
        self.distance = np.float64(0.)
        for (start, end), middle in zip(itertools.pairwise(self.path_vertices),
                                        self.path_halfedges):
            start_point, double_boundary, boundary, \
                interior_shared, interior_unshared, end_point \
                = self._get_point_locations(start, middle, end)
            self._start_points.append(start_point)
            self._double_boundary.append(double_boundary)
            self._boundary.append(boundary)
            self._interior_shared.append(interior_shared)
            self._interior_unshared.append(interior_unshared)
            self._end_points.append(end_point)
            self.distance += np.linalg.norm(start_point - end_point)

    def _reverse_part(self,
                      start_point: npt.NDArray[np.float64],
                      double_boundary: typing.List[typing.Tuple[
                          dcelmesh.Mesh.Halfedge,
                          npt.NDArray[np.float64],
                          npt.NDArray[np.float64]
                      ]],
                      boundary: typing.List[typing.Tuple[
                          dcelmesh.Mesh.Halfedge,
                          npt.NDArray[np.float64],
                          npt.NDArray[np.float64],
                          npt.NDArray[np.float64]
                      ]],
                      interior_shared: typing.List[typing.Tuple[
                          dcelmesh.Mesh.Halfedge,
                          npt.NDArray[np.float64],
                          npt.NDArray[np.float64],
                          npt.NDArray[np.float64],
                          npt.NDArray[np.float64]
                      ]],
                      interior_unshared: typing.List[typing.Tuple[
                          dcelmesh.Mesh.Halfedge,
                          npt.NDArray[np.float64],
                          npt.NDArray[np.float64],
                          npt.NDArray[np.float64],
                          npt.NDArray[np.float64]
                      ]],
                      end_point: npt.NDArray[np.float64]) \
            -> None:
        """
        Compute the partials for a geodesic not passing through saddles.

        In other words, the only mesh points the geodesic path should
        coincide with are the endpoints.
        """
        d_geodesic = np.linalg.norm(end_point - start_point)

        for halfedge, pu, pv in double_boundary:
            u = halfedge.origin()
            v = halfedge.destination()
            edge = halfedge.edge()
            partial_edge = 1.

            self.dif_distance[u.index()] += partial_edge \
                * self.dif_edge_lengths[edge.index()][u.index()]
            self.dif_distance[v.index()] += partial_edge \
                * self.dif_edge_lengths[edge.index()][v.index()]

        for halfedge, pu, pv, pw in boundary:
            u = halfedge.origin()
            v = halfedge.destination()
            edge = halfedge.edge()

            # Use bad names here and elsewhere to avoid lines becoming
            # too long and (even more) unreadable
            sw = start_point - pw
            ew = end_point - pw
            uw = pu - pw
            vw = pv - pw
            vu = pv - pu

            partial_edge = np.abs(np.linalg.norm(vu) * np.cross(sw, ew)
                                  / (d_geodesic * np.cross(uw, vw)))

            self.dif_distance[u.index()] += partial_edge \
                * self.dif_edge_lengths[edge.index()][u.index()]
            self.dif_distance[v.index()] += partial_edge \
                * self.dif_edge_lengths[edge.index()][v.index()]

        for halfedge, pu, pv, pw, px in interior_shared:
            u = halfedge.origin()
            v = halfedge.destination()
            edge = halfedge.edge()

            su = start_point - pu
            wu = pw - pu
            wv = pw - pv
            eu = end_point - pu
            xu = px - pu
            xv = px - pv
            vu = pv - pu

            partial_edge = -np.linalg.norm(vu) * np.cross(eu, su) \
                * (1. / np.cross(wu, wv) + 1. / np.cross(xv, xu)) \
                / (d_geodesic * (1. + np.cross(xu, wu) / np.cross(wv, xv)))

            self.dif_distance[u.index()] += partial_edge \
                * self.dif_edge_lengths[edge.index()][u.index()]
            self.dif_distance[v.index()] += partial_edge \
                * self.dif_edge_lengths[edge.index()][v.index()]

        for halfedge, pu, pv, pw, px in interior_unshared:
            u = halfedge.origin()
            v = halfedge.destination()
            edge = halfedge.edge()

            wu = pw - pu
            wv = pw - pv
            su = start_point - pu
            sv = start_point - pv
            sw = start_point - pw
            xu = px - pu
            xv = px - pv
            eu = end_point - pu
            ev = end_point - pv
            ex = end_point - px
            vu = pv - pu

            partial_edge = np.linalg.norm(vu) * (
                (
                    np.cross(wv, sw) / np.cross(wu, wv)
                    * (
                        (1. - sv @ ev / (sv @ sv)) / np.cross(sv, ev)
                        + (1. - sv @ su / (sv @ sv)) / np.cross(su, sv)
                    )
                ) + (
                    np.cross(xu, ex) / np.cross(xv, xu)
                    * (
                        (1. - eu @ su / (eu @ eu)) / np.cross(eu, su)
                        + (1. - eu @ ev / (eu @ eu)) / np.cross(ev, eu)
                    )
                )
                - 1. / np.cross(su, sv)
                - 1. / np.cross(ev, eu)
            ) / (d_geodesic * (
                1. / np.cross(sv, ev)
                + 1. / np.cross(eu, su)
            ))

            self.dif_distance[u.index()] += partial_edge \
                * self.dif_edge_lengths[edge.index()][u.index()]
            self.dif_distance[v.index()] += partial_edge \
                * self.dif_edge_lengths[edge.index()][v.index()]

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

        self.dif_edge_lengths = [{} for _ in self._topology.edges()]
        self.dif_distance = {}

        # Compute the partials of edge lengths first
        for element in itertools.chain(
            itertools.chain(*self._double_boundary),
            itertools.chain(*self._boundary),
            itertools.chain(*self._interior_shared),
            itertools.chain(*self._interior_unshared)
        ):
            edge = element[0].edge()
            if self.dif_edge_lengths[edge.index()]:
                continue
            u, v = edge.vertices()
            pu = self._coordinates[u.index()]
            pv = self._coordinates[v.index()]
            edge_length = self.edge_lengths[edge.index()]
            self.dif_edge_lengths[edge.index()][u.index()] \
                = (pu - pv) @ self._partials[u.index()] / edge_length
            self.dif_edge_lengths[edge.index()][v.index()] \
                = (pv - pu) @ self._partials[v.index()] / edge_length

            # Set up for accumulation
            self.dif_distance[u.index()] = np.float64(0.)
            self.dif_distance[v.index()] = np.float64(0.)

        # Finally, we actually do the accumulation
        for start_point, double_boundary, boundary, \
                interior_shared, interior_unshared, end_point \
                in zip(self._start_points,
                       self._double_boundary,
                       self._boundary,
                       self._interior_shared,
                       self._interior_unshared,
                       self._end_points):
            self._reverse_part(start_point, double_boundary, boundary,
                               interior_shared, interior_unshared, end_point)
