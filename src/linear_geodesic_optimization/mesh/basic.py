"""Module containing an struct-style mesh representing a manifold."""

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

    def __init__(self, topology: dcelmesh.Mesh,
                 coordinates: npt.NDArray[np.float64]):
        self._updates = 0
        self._topology = topology
        self._coordinates = coordinates

    def get_topology(self) -> dcelmesh.Mesh:
        """Return the topology of the mesh."""
        return self._topology

    def get_coordinates(self) -> npt.NDArray[np.float64]:
        """
        Return the coordinates of the vertices of the mesh.

        For efficiency, this returns a |V| by 3 array.
        """
        return self._coordinates

    def get_parameters(self) -> npt.NDArray[np.float64]:
        """
        Return the parameters of the vertices of this mesh.

        The output is ordered in the same way as the output of
        `Mesh.get_coordinates`.
        """
        return self._coordinates[:,2]

    def set_parameters(self, parameters: npt.NDArray[np.float64]) \
            -> npt.NDArray[np.float64]:
        """
        Set the parameters of the vertices of this mesh.

        The input to this function should be ordered in the same way as
        the output of `Mesh.get_coordinates`. This function returns the
        resulting parameters, which should be treated as read only.
        """
        self._updates += 1
        self._coordinates[:,2] = parameters
        return self._coordinates

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

        We assume the mesh is parameterized by |V| scalars, each of
        which affects exactly one vertex. For efficiency, this returns a
        |V| by 3 array. The output of this function should be treated as
        read only.
        """
        partials = np.zeros(self._coordinates.size)
        partials[:,2] = 1.
        return partials
