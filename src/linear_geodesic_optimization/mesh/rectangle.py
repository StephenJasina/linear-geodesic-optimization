"""Module containing an implementation of a rectangluar mesh."""

import typing

import dcelmesh
import numpy as np
import numpy.typing as npt

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
        [-`scale` / 2, `scale` / 2]^2.

        The z-coordinates are bounded between 0 and `extent`.
        """
        self._width: int = width
        self._height: int = height
        self._scale: np.float64 = scale

        self.center = None
        self.scale_factor = None

        self._topology: dcelmesh.Mesh = self._get_topology()
        self._coordinates: npt.NDArray[np.float64] = self._get_coordinates()

        self._extent: np.float64 = extent

        self._parameters: npt.NDArray[np.float64] = np.zeros(width * height)

        self._updates: int = 0

        self._partials: npt.NDArray[np.float64] = np.zeros((width * height, 3))
        self._partials[:, 2] = 1.

    def _get_topology(self) -> dcelmesh.Mesh:
        topology = dcelmesh.Mesh(self._width * self._height)

        # Add faces cell-by-cell
        for i in range(self._width - 1):
            for j in range(self._height - 1):
                v00 = i * self._height + j            # Bottom-left
                v01 = i * self._height + j + 1        # Top-left
                v10 = (i + 1) * self._height + j      # Bottom-right
                v11 = (i + 1) * self._height + j + 1  # Top-right

                # Add the top-left triangle
                topology.add_face([v00, v11, v01])

                # Add the bottom-right triangle
                topology.add_face([v00, v10, v11])

        return topology

    def _get_coordinates(self) -> npt.NDArray[np.float64]:
        coordinates = np.zeros((self._width * self._height, 2))

        # The vertices are ordered lexicographically as (x, y)
        for i in range(self._width):
            for j in range(self._height):
                coordinates[i * self._height + j:] = [
                    (i / (self._width - 1) - 0.5) * self._scale,
                    (j / (self._height - 1) - 0.5) * self._scale
                ]

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
        if not np.array_equal(self._parameters, z):
            self._parameters = np.copy(z)
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
        def is_on_fat_edge(
            u: npt.NDArray[np.float64],
            v: npt.NDArray[np.float64],
            r: npt.NDArray[np.float64],
            epsilon: np.float64
        ) -> bool:
            """
            Find whether a point is close to a line segment.

            In detail, determine whether `r` is within `epsilon` of the
            line segment between `u` and `v`.
            """
            # Only care about the first two coordinates
            u = u[:2]
            v = v[:2]
            r = r[:2]

            ru = r - u
            rv = r - v
            uv = u - v

            if ru @ uv <= 0 and rv @ uv >= 0:
                if uv @ uv == 0:
                    return ru @ ru < epsilon**2
                return rv @ rv - (uv @ rv)**2 / (uv @ uv) < epsilon**2
            else:
                return ru @ ru < epsilon**2 or rv @ rv < epsilon**2

        return [[self._topology.get_vertex(index)
                 for index in range(self._coordinates.shape[0])
                 if is_on_fat_edge(vertices[e1], vertices[e2],
                                   self._coordinates[index, :], epsilon)]
                for (e1, e2) in edges]

    def nearest_vertex(self, coordinate: npt.NDArray[np.float64]) \
            -> dcelmesh.Mesh.Vertex:
        """
        Find the closest mesh vertex to the input coordinate pair.

        Distance is measured solely by x-y coordinates. We assume the
        input lies in the mesh's support [-`scale` / 2, `scale` / 2]^2.
        """
        i = round((coordinate[0] / self._scale + 0.5) * (self._width - 1))
        j = round((coordinate[1] / self._scale + 0.5) * (self._height - 1))
        return self._topology.get_vertex(i * self._height + j)

    def map_coordinates_to_support(
            self,
            coordinates: npt.NDArray[np.float64],
            scale_factor: np.float64 = np.float64(1.)
    ) -> npt.NDArray[np.float64]:
        """
        Rescale coordinates so that they lie within the mesh support.

        In more detail, convert an array of (x, y) pairs into an array
        of new coordinates that have been scaled to lie centered in the
        support. The `scale_factor` parameter determines what proportion
        of the support is used (0.45 means at most 45% of the width and
        45% of the height is used).
        """
        # Need this check to avoid out-of-bounds errors if coordinates
        # is empty
        if coordinates.size == 0:
            return np.array([])

        coordinates = coordinates[:, :2]

        coordinates_min = np.amin(coordinates, axis=0)
        coordinates_max = np.amax(coordinates, axis=0)
        divisor = coordinates_max - coordinates_min
        divisor[np.where(divisor == 0.)] = 1.
        divisor = np.amax(divisor)

        self.center = (coordinates_min + coordinates_max) / 2
        self.scale_factor = scale_factor / divisor

        return self._scale * self.scale_factor * (coordinates - self.center)

    def get_support_area(self) -> np.float64:
        """
        Return the support area of this mesh.

        This is a simple length-times-width computation, where both of
        them is given by the `scale` parameter passed during
        initialization.
        """
        return self._scale**2

    def get_width(self) -> int:
        """Return the number of vertices horizontally in this mesh."""
        return self._width

    def get_height(self) -> int:
        """Return the number of vertices vertically in this mesh."""
        return self._height
