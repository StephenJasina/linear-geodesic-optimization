"""Module containing an implementation of a rectangluar mesh."""

import itertools
import typing

import dcelmesh
import numpy as np
import numpy.typing as npt

from linear_geodesic_optimization import geometry
import linear_geodesic_optimization.mesh.mesh


class Mesh(linear_geodesic_optimization.mesh.mesh.Mesh):
    """
    Representation of a mesh that is "approximately" a rectangle.

    In particular, projecting the vertices of the mesh so that their
    z-coordinates are 0 will yield a rectangle. The mesh itself looks
    like a grid where each cell has been cut by its major diagonal.
    """

    def __init__(self, width: int, height: int,
                 scale: np.float64 = np.float64(1.),
                 extent: np.float64 = np.float64(np.inf)):
        """
        Create a rectangular mesh.

        Topologically, the mesh will have `width` vertices horizontally
        and `height` vertices vertically. The edges are such that the
        mesh is composed of square cells with a major diagonal.

        For the coordinate computations, the support of the mesh is
        [0, `scale`]^2.

        The z-coordinates are bounded between 0 and `extent`.
        """
        self._width: int = width
        self._height: int = height
        self._scale: np.float64 = scale

        self._topology: dcelmesh.Mesh = self._get_topology()
        """`dcelmesh.Mesh` encoding of this mesh's connectivity."""
        self._coordinates: npt.NDArray[np.float64] = self._get_coordinates()
        """x-y coordinates of this mesh's points."""

        self._extent: np.float64 = extent

        self._parameters: npt.NDArray[np.float64] = np.zeros(width * height)
        """Parameters controlling the z-coordinates of this mesh."""

        self._updates: int = 0

        self._partials: npt.NDArray[np.float64] = np.zeros((width * height, 3))
        self._partials[:, 2] = 1.

        self._trim_mapping = list(range(self._topology.n_vertices))

    def get_width(self) -> int:
        return self._width

    def get_height(self) -> int:
        return self._width

    def get_scale(self) -> float:
        return self._scale

    def _get_topology(self) -> dcelmesh.Mesh:
        topology = dcelmesh.Mesh(self._width * self._height)
        vertices = list(topology.vertices())

        # Add faces cell-by-cell
        for i in range(self._width - 1):
            for j in range(self._height - 1):
                v00 = i * self._height + j            # Bottom-left
                v01 = i * self._height + j + 1        # Top-left
                v10 = (i + 1) * self._height + j      # Bottom-right
                v11 = (i + 1) * self._height + j + 1  # Top-right

                # Add the top-left triangle
                topology.add_face([vertices[v00], vertices[v11], vertices[v01]])

                # Add the bottom-right triangle
                topology.add_face([vertices[v00], vertices[v10], vertices[v11]])

        return topology

    def _get_coordinates(self) -> npt.NDArray[np.float64]:
        coordinates = np.zeros((self._width * self._height, 2))

        # The vertices are ordered lexicographically as (x, y)
        for i in range(self._width):
            for j in range(self._height):
                coordinates[i * self._height + j, 0] = i / (self._width - 1) * self._scale
                coordinates[i * self._height + j, 1] = j / (self._height - 1) * self._scale

        return coordinates

    def get_topology(self) -> dcelmesh.Mesh:
        """Return the topology of the mesh."""
        return self._topology

    def get_coordinates(self) -> npt.NDArray[np.float64]:
        """
        Return the coordinates of the vertices of the mesh.

        For efficiency, this returns a |V| by 3 array.
        """
        if self._extent == np.inf:
            z = self._parameters
        else:
            z = self._extent / (1 + np.exp(-self._parameters))
        return np.concatenate((self._coordinates,
                               np.reshape(z, (-1, 1))), axis=1)

    def get_parameters(self) -> npt.NDArray[np.float64]:
        """
        Return the z-coordinates of the vertices of this mesh.

        The output is ordered in the same way as the output of
        `Mesh.get_coordinates`.
        """
        return self._parameters

    def set_parameters(self, z: npt.NDArray[np.float64]) \
            -> npt.NDArray[np.float64]:
        """
        Set the z-coordinates of the vertices of this mesh.

        The input to this function should be ordered in the same way as
        the output of `Mesh.get_coordinates`. This function returns the
        resulting parameters, which should be treated as read only.
        """
        z = np.copy(z)
        if z.shape != self._parameters.shape:
            raise ValueError()

        # TODO: Is this too slow?
        if not np.array_equal(self._parameters, z):
            self._parameters = z
            self._updates += 1
        return self._parameters

    def get_updates(self) -> int:
        """
        Return the number of calls to `Mesh.set_parameters`.

        This function is an easy (O(1)) way to determine whether the
        mesh has been updated.
        """
        return self._updates

    def get_partials(self) -> npt.NDArray[np.float64]:
        """
        Return the partials of each of the vertices' parameters.

        If the `extent` the mesh was initialized is infinity, then each
        partial is just a unit vertex pointing along the z-axis.

        Otherwise, the partials are determined according to the logistic
        (sigmoid) function of the corresponding parameter, scaled so
        that the output lies between 0 and `extent`.

        The returned value should be treated as read only.
        """
        if self._extent != np.inf:
            self._partials[:, 2] = self._extent * np.exp(self._parameters) \
                / (1 + np.exp(self._parameters))**2
        return self._partials

    def get_fat_edges(
        self,
        vertices: npt.NDArray[np.float64],
        edges: typing.List[typing.Tuple[int, int]],
        epsilon: np.float64
    ) -> typing.List[typing.List[dcelmesh.Mesh.Vertex]]:
        """
        Find fattened versions of edges when mapped onto this mesh.

        For a list of edges in a graph embedded in our mesh (represented
        as pairs of indices into `vertices`) and a width `epsilon` > 0,
        return a list of lists of vertices in our mesh. Each list of
        vertices corresponds to a fattened edge, where the vertices are
        those within a distance of epsilon of the original edge when
        everything is projected to the x-y plane.
        """
        return [
            [
                vertex
                for vertex in self._topology.vertices()
                if geometry.distance_to_line_segment(
                    v1, v2, self._coordinates[vertex.index, :2]
                ) < epsilon
            ]
            for (v1_index, v2_index) in edges
            for v1 in (vertices[v1_index, :2],)
            for v2 in (vertices[v2_index, :2],)
        ]

    def nearest_vertex(self, coordinate: npt.NDArray[np.float64]) \
            -> dcelmesh.Mesh.Vertex:
        """
        Find the closest mesh vertex to the input coordinate pair.

        Distance is measured solely by x-y coordinates.

        TODO: Note that this implementation is rather inefficient. This
        can possibly be improved using some modular arithmetic.
        """
        return self._topology.get_vertex(np.argmin(np.linalg.norm(
            self._coordinates - coordinate, axis=1
        )))

    def map_coordinates_to_support(
            self,
            coordinates: npt.NDArray[np.float64],
            scale_factor: np.float64 = np.float64(1.),
            bounding_box: typing.Optional[
                typing.Tuple[np.float64, np.float64, np.float64, np.float64]
            ] = None
    ) -> npt.NDArray[np.float64]:
        """
        Rescale coordinates so that they lie within the mesh support.

        In more detail, convert an array of (x, y) pairs into an array
        of new coordinates that have been scaled to lie centered in the
        support. The `scale_factor` parameter determines what proportion
        of the support is used (0.45 means at most 45% of the width and
        45% of the height is used).

        If a bounding box is passed in, use that to determine
        [long_min, long_max, lat_min, lat_max].
        """
        # Need this check to avoid out-of-bounds errors if coordinates
        # is empty
        if coordinates.size == 0:
            return np.array([])

        coordinates = coordinates[:, :2]

        if bounding_box is None:
            coordinates_min = np.amin(coordinates, axis=0)
            coordinates_max = np.amax(coordinates, axis=0)
        else:
            coordinates_min = np.array([bounding_box[0], bounding_box[2]])
            coordinates_max = np.array([bounding_box[1], bounding_box[3]])

        divisor = coordinates_max - coordinates_min
        divisor[np.where(divisor == 0.)] = 1.
        divisor = np.amax(divisor)

        return self._scale * (scale_factor * (coordinates - (coordinates_min + coordinates_max) / 2.) / divisor + 0.5)

    def trim_to_graph(
        self,
        vertices: npt.NDArray[np.float64],
        edges: typing.List[typing.Tuple[int, int]],
        epsilon: np.float64
    ) -> None:
        if np.isposinf(epsilon):
            return

        fat_edges = self.get_fat_edges(vertices, edges, epsilon)
        included_vertex_indices = set(
            vertex.index
            for fat_edge in fat_edges
            for vertex in fat_edge
        )
        self.trim_to_set(included_vertex_indices)

    def trim_to_set(self, indices: typing.Iterable[int]):
        included_vertex_indices = set(indices)
        excluded_vertices = [
            vertex
            for vertex in self._topology.vertices()
            if vertex.index not in included_vertex_indices
        ]
        for vertex in excluded_vertices:
            self._topology.remove_vertex(vertex)

        mapping = self._topology.reindex()
        self._coordinates = self._coordinates[mapping, :]
        self._parameters = self._parameters[mapping]
        self._partials = self._partials[mapping, :]
        self._trim_mapping = [
            self._trim_mapping[index]
            for index in mapping
        ]

    def restore_removed_vertices(self):
        self._topology = self._get_topology()
        self._coordinates = self._get_coordinates()
        self._parameters = np.zeros(self._width * self._height)
        self._partials: npt.NDArray[np.float64] = np.zeros((self._width * self._height, 3))
        self._partials[:, 2] = 1.
        self._trim_mapping = list(range(self._topology.n_vertices))

    def get_trim_mapping(self) -> typing.List[int]:
        return self._trim_mapping

    def add_vertex_at_coordinates(self, p, epsilon=0.) -> dcelmesh.Mesh.Vertex:
        """
        Add a vertex into the mesh at the given x-y coordinates.
        """
        # Check whether `p` is very close to one of the vertices `a`
        for va in self._topology.vertices():
            a = self._coordinates[va.index]
            if np.linalg.norm(p - a) <= epsilon * self._scale:
                # If `p` and `a` are sufficiently close, there is
                # nothing to do
                return va

        # Check whether `p` is very close to one of the edges `a`-`b`
        # We first find the closest edge, which is slow, but will
        # produce better results in the case that epsilon is too large
        closest_edge = None
        closest_edge_distance = np.inf
        for edge in self._topology.edges():
            va, vb = edge.vertices()
            a = self._coordinates[va.index]
            b = self._coordinates[vb.index]
            distance = geometry.distance_to_line_segment(a, b, p)
            if distance < closest_edge_distance:
                closest_edge = edge
                closest_edge_distance = distance
        if closest_edge_distance <= epsilon:
            # If `p` is nearly on one of the edges, add it an ensure
            # the mesh remains a triangulation
            va, vb = closest_edge.vertices()
            a = self._coordinates[va.index]
            b = self._coordinates[vb.index]
            ba = np.append(b - a, [self._parameters[vb.index] - self._parameters[va.index]], 0)
            ba = ba / np.linalg.norm(ba[:2])
            pa = p - a
            lambda_ = pa @ ba[:2]
            z = lambda_ * ba[2] + self._parameters[va.index]

            vp = self._topology.add_vertex_into_edge(closest_edge)
            self._coordinates = np.append(self._coordinates, p.reshape((1, 2)), 0)
            self._parameters = np.append(self._parameters, [z], 0)
            self._partials = np.append(self._partials, [[0., 0., 1.]], 0)
            return vp

        # Add `p` to the face it lies in, if such a face exists
        for face in self._topology.faces():
            va, vb, vc = face.vertices()
            a = self._coordinates[va.index]
            b = self._coordinates[vb.index]
            c = self._coordinates[vc.index]
            if geometry.is_in_triangle(a, b, c, p):
                # Treat a as the origin and use Graham-Schmidt
                ba = np.append(b - a, [self._parameters[vb.index] - self._parameters[va.index]], 0)
                ba = ba / np.linalg.norm(ba[:2])
                ca = np.append(c - a, [self._parameters[vc.index] - self._parameters[va.index]], 0)
                ca = ca - (ca[:2] @ ba[:2]) * ba
                ca = ca / np.linalg.norm(ca[:2])
                pa = p - a
                lambda_b = pa @ ba[:2]
                lambda_c = pa @ ca[:2]
                z = lambda_b * ba[2] + lambda_c * ca[2] + self._parameters[va.index]

                vp = self._topology.add_vertex_into_face(face)
                self._coordinates = np.append(self._coordinates, p.reshape((1, 2)), 0)
                self._parameters = np.append(self._parameters, [z], 0)
                self._partials = np.append(self._partials, [[0., 0., 1.]], 0)
                return vp

        raise ValueError('Point does not lie inside mesh support')

    def remove_added_vertices(self) -> None:
        n_vertices = self._width * self._height
        added_vertices = reversed(list(itertools.islice(
            self._topology.vertices(),
            n_vertices, None
        )))
        for vertex in added_vertices:
            self._topology.remove_vertex_out_of_face(vertex)
        self._topology.reindex()

        self._coordinates = self._coordinates[:n_vertices,:]
        self._parameters = self._parameters[:n_vertices]
        self._partials = self._partials[:n_vertices,:]
