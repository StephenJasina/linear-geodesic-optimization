"""Module containing an interface for a mesh representing a manifold."""

import typing

import dcelmesh
import numpy as np


class Mesh:
    """
    A triangular mesh representing a manifold.

    The mesh can be decomposed into its underlying topology and the
    coordinates of its vertices. Furthermore, each vertex can be thought
    of as being determined by a single parameter.
    """

    def get_topology(self) -> dcelmesh.Mesh:
        """Return the topology of the mesh."""
        pass

    def get_coordinates(self) -> np.ndarray:
        """
        Return the coordinates of the vertices of the mesh.

        For efficiency, this returns a |V| by 3 array.
        """
        raise NotImplementedError

    def get_parameters(self) -> np.ndarray:
        """
        Return the parameters of the vertices of this mesh.

        The output is ordered in the same way as the output of
        `Mesh.get_coordinates`.
        """
        raise NotImplementedError

    def set_parameters(self, parameters: np.ndarray) -> np.ndarray:
        """
        Set the parameters of the vertices of this mesh.

        The input to this function should be ordered in the same way as
        the output of `Mesh.get_coordinates`. This function returns the
        resulting parameters, which should be treated as read only.
        """
        raise NotImplementedError

    def updates(self) -> int:
        """
        Return the number of calls to `Mesh.set_parameters`.

        This function is an easy (O(1)) way to determine whether the
        mesh has been updated.
        """
        raise NotImplementedError

    def get_partials(self) -> np.ndarray:
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
        vertices: typing.List[typing.List[float]],
        edges: typing.List[typing.Tuple[int, int]],
        epsilon: float
    ) -> typing.List[typing.List[dcelmesh.Mesh.Vertex]]:
        """
        Find fattened versions of edges when mapped onto this mesh.

        For a list of edges in a graph embedded in our mesh (represented
        as pairs of indices into `vertices`) and a width `epsilon` > 0,
        return a list of lists of vertices in our mesh. Each list of
        vertices corresponds to a fattened edge.
        """
        raise NotImplementedError

    def get_support_area(self) -> float:
        """
        Return the area of the support of the mesh.

        For example, if the support is the unit sphere, then this will
        return 4 * pi.
        """
        raise NotImplementedError
