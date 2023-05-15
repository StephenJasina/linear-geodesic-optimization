import dcelmesh
import numpy as np

import linear_geodesic_optimization.mesh.mesh


class Mesh(linear_geodesic_optimization.mesh.mesh.Mesh):
    '''
    Representation of a mesh that is "approximately" a rectangle. In
    particular, projecting the vertices of the mesh so that their z-coordinates
    are 0 will yield a rectangle. The mesh itself looks like a grid where each
    cell has been cut by its major diagonal.
    '''

    def __init__(self, width, height, scale=1., extent=np.inf):
        self._width = width
        self._height = height
        self._scale = scale

        self._topology, self._coordinates = self._get_mesh()

        self._extent = np.inf

        self._partials = np.zeros((width * height, 3))
        self._partials[:, 2] = 1.
        self._parameters = np.zeros(width * height)

        self._updates = 0

    def _get_mesh(self):
        topology = dcelmesh.Mesh(self._width * self._height)
        coordinates = np.zeros((self._width * self._height, 2))

        # The vertices are ordered lexicographically as (x, y)
        for i in range(self._width):
            for j in range(self._height):
                coordinates[i * self._height + j:] = [
                    (i / (self._width - 1) - 0.5) * self._scale,
                    (j / (self._height - 1) - 0.5) * self._scale
                ]

        # Add edges and faces cell-by-cell
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

        return topology, coordinates

    def get_topology(self):
        return self._topology

    def get_partials(self):
        if self._extent != np.inf:
            self._partials = self._extent * np.exp(self._parameters) \
                / (1 + np.exp(-self._parameters))**2
        return self._partials

    def get_coordinates(self):
        if self._extent == np.inf:
            z = self._parameters
        else:
            z = self._extent / (1 + np.exp(-self._parameters))
        return np.concatenate((self._coordinates,
                               np.reshape(z, (-1, 1))), axis=1)

    def get_parameters(self):
        return self._parameters

    def set_parameters(self, z):
        if not np.array_equal(self._parameters, z):
            self._parameters = np.copy(z)
            self._updates += 1
        self._parameters

    def updates(self):
        return self._updates

    def get_fat_edges(self, vertices, edges, epsilon=None):
        def is_on_fat_edge(u, v, r, epsilon):
            '''
            Determine whether `r` is within `epsilon` of the line segment
            between `u` and `v`.
            '''

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

        return [[i for i in range(self._coordinates.shape[0])
                 if is_on_fat_edge(vertices[e1], vertices[e2],
                                   self._coordinates[i, :], epsilon)]
                for (e1, e2) in edges]

    def nearest_vertex_index(self, v):
        '''
        Find the index of the vertex whose (x, y) coordinate pair is closest to
        the input coordinate pair. We assume the input lies in
        [-scale / 2, scale / 2]^2.
        '''

        i = round((v[0] / self._scale + 0.5) * (self._width - 1))
        j = round((v[1] / self._scale + 0.5) * (self._height - 1))
        return i * self._height + j

    def map_coordinates_to_support(self, coordinates, scale_factor=1.):
        '''
        Convert a list of (x, y, z) triples into a list of new coordinates that
        have been scaled to lie centered in the support. The `scale_factor`
        parameter determines what proportion of the support is used (0.45
        means 45% of the width and 45% of the height is used).
        '''

        # Need this check to avoid out-of-bounds errors if coordinates is empty
        if not coordinates:
            return []

        coordinates = np.array(coordinates)[:, :2]

        coordinates_min = np.amin(coordinates, axis=0)
        coordinates_max = np.amax(coordinates, axis=0)
        divisor = coordinates_max - coordinates_min
        divisor[np.where(divisor == 0.)] = 1.
        divisor = np.amax(divisor)

        coordinates = coordinates - (coordinates_min + coordinates_max) / 2
        coordinates = coordinates / divisor

        return list(self._scale * scale_factor * coordinates)

    def get_support_area(self):
        return self._scale**2

    def get_width(self):
        return self._width

    def get_height(self):
        return self._height
