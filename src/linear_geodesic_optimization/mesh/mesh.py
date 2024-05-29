"""Module containing an interface for a mesh representing a manifold."""

import typing

import dcelmesh
import numpy as np
import numpy.typing as npt


class Mesh:
    """
    A triangular mesh representing a manifold.

    The mesh can be decomposed into its underlying topology and the
    coordinates of its vertices. Furthermore, each vertex can be thought
    of as being determined by a single parameter.
    """

    def get_topology(self) -> dcelmesh.Mesh:
        """Return the topology of the mesh."""
        raise NotImplementedError

    def get_coordinates(self) -> npt.NDArray[np.float64]:
        """
        Return the coordinates of the vertices of the mesh.

        For efficiency, this returns a |V| by 3 array.
        """
        raise NotImplementedError

    def get_parameters(self) -> npt.NDArray[np.float64]:
        """
        Return the parameters of the vertices of this mesh.

        The output is ordered in the same way as the output of
        `Mesh.get_coordinates`.
        """
        raise NotImplementedError

    def set_parameters(self, parameters: npt.NDArray[np.float64]) \
            -> npt.NDArray[np.float64]:
        """
        Set the parameters of the vertices of this mesh.

        The input to this function should be ordered in the same way as
        the output of `Mesh.get_coordinates`. This function returns the
        resulting parameters, which should be treated as read only.
        """
        raise NotImplementedError

    def get_updates(self) -> int:
        """
        Return the number of calls to `Mesh.set_parameters`.

        This function is an easy (O(1)) way to determine whether the
        mesh has been updated.
        """
        raise NotImplementedError

    def get_partials(self) -> npt.NDArray[np.float64]:
        """
        Return the partials of each of the vertices' parameters.

        We assume the mesh is parameterized by |V| scalars, each of
        which affects exactly one vertex. For efficiency, this returns a
        |V| by 3 array. The output of this function should be treated as
        read only.
        """
        raise NotImplementedError

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
        vertices corresponds to a fattened edge.
        """
        raise NotImplementedError

    def nearest_vertex(self, coordinate: npt.NDArray[np.float64]) \
            -> dcelmesh.Mesh.Vertex:
        """
        Find the closest mesh vertex to the input coordinates.

        This function discretizes the mesh by giving a direct
        conversion from continuous coordinates to discrete vertices.
        """
        raise NotImplementedError

    def map_latencies_to_mesh(
        self,
        network_vertices: typing.List[typing.Tuple[np.float64, np.float64]],
        network_latencies: typing.List[typing.Tuple[typing.Tuple[int, int],
                                                    np.float64]]
    ) -> typing.List[typing.Tuple[typing.Tuple[int, int], np.float64]]:
        """
        Convert latencies from a network to a mesh.

        As input, take a mesh, a list of coordinates, and latencies (as
        returned by `read_graphml`). Return a list of latencies stored along
        with the pair of corresponding mesh vertex indices, in the form
        `((i, j), latency)`.

        This format is used for space efficiency (not many points on the
        mesh are expected to have measured latencies).
        """
        latencies: typing.List[typing.Tuple[typing.Tuple[int, int],
                                            np.float64]] = []

        # Can't do a dict comprehension since multiple vertices could
        # map to the same mesh point
        for (i, j), latency in network_latencies:
            i_key = self.nearest_vertex(network_vertices[i]).index
            j_key = self.nearest_vertex(network_vertices[j]).index
            latencies.append(((i_key, j_key), latency))

        return latencies

    def get_support_area(self) -> np.float64:
        """
        Return the area of the support of the mesh.

        For example, if the support is the unit sphere, then this will
        return 4 * pi.
        """
        raise NotImplementedError
