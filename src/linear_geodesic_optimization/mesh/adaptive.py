import numpy as np

from linear_geodesic_optimization.mesh import mesh

class Mesh(mesh.Mesh):
    '''
    Representation of a mesh that refines itself so that, given a set of
    points, each cell contains at most one point.
    '''

    def __init__(self, points, epsilon):
        self._points = points
        self._epsilon = epsilon

        # Internally store vertices in 2 dimensions
        self._vertices, self._edges, self._faces, self._c \
            = self._initial_mesh()
        self._parameters = np.zeros(self._vertices.shape[0])
        self._partials = np.zeros((self._vertices.shape[0], 3))
        self._partials[:,2] = 1.
        self._updates = 0

    def _split_face(self, index):
        raise NotImplementedError

    def _initial_mesh(self):
        vertices = [[0, 0], [1, 0], [0, 1], [1, 1]]
        edges = [[1, 3], [3], [0], [0, 2]]
        faces = [(0, 1, 3), (0, 3, 2)]
        c = {(0, 1): 3, (0, 3): 2, (1, 3): 0, (2, 0): 3, (3, 0): 1, (3, 2): 0}

        return np.array(vertices), edges, faces, c

    def get_partials(self):
        return np.copy(self._partials)

    def get_vertices(self):
         return np.concatenate((self._vertices,
                               np.reshape(self._parameters, (-1, 1))), axis=1)

    def get_edges(self):
        return self._edges

    def get_faces(self):
        return self._faces

    def get_boundary_vertices(self):
        return set(u for u, _ in self.get_boundary_edges())

    def get_boundary_edges(self):
        c = self.get_c()
        return set((v, u) for u, v in c if (v, u) not in c)

    def get_c(self):
        return self._c

    def get_parameters(self):
        return np.copy(self._parameters)

    def set_parameters(self, parameters):
        if not np.array_equal(self._parameters, parameters):
            self._parameters = np.copy(parameters)
            self._updates += 1
        return np.copy(parameters)

    def updates(self):
        return self._updates

    def get_fat_edges(self, vertices, edges, epsilon):
        raise NotImplementedError

    def get_epsilon(self):
        return self._epsilon

    def get_support_area(self):
        return 1.

    def map_coordinates_to_support(self, coordinates, scale_factor=0.45):
        '''
        Convert a list of (x, y, z) triples into a list of new coordinates that
        have been scaled to lie centered in the unit square. The `scale_factor`
        parameter determines what proportion of the unit square is used (0.45
        means 45% of the width and 45% of the height is used).
        '''

        # Need this check to avoid out-of-bounds errors if coordinates is empty
        if not coordinates:
            return []

        coordinate_min = coordinates[0][0]
        coordinate_max = coordinates[0][0]

        # Find scaling factors so that the x range and y range are both [0, 1].
        # If no such scaling factor exists (i.e., the x-coordinate or
        # y-coordinate is constant), set it to some arbitrary value (in this
        # case, just set it to 1.)
        for x, y in coordinates:
            coordinate_min = min(coordinate_min, x, y)
            coordinate_max = max(coordinate_max, x, y)
        divisor = coordinate_max - coordinate_min
        divisor = 1. if divisor == 0. else divisor

        return [np.array([((x - coordinate_min) / divisor - 0.5) * scale_factor,
                          ((y - coordinate_min) / divisor - 0.5) * scale_factor,
                          0])
                for x, y in coordinates]
