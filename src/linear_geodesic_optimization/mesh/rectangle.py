import numpy as np

from linear_geodesic_optimization.mesh import mesh

class Mesh(mesh.Mesh):
    def __init__(self, width, height):
        self._width = width
        self._height = height
        self._grid, self._edges, self._faces, self._c \
            = self._initial_mesh(width, height)
        self._partials = np.zeros((self._grid.shape[0], 3))
        self._partials[:,2] = 1.
        self._z = np.zeros(self._grid.shape[0])
        self._updates = 0

    def _initial_mesh(self, width, height):
        vertices = np.zeros((width * height, 3))
        for i in range(width):
            for j in range(height):
                vertices[i*height+j:] = np.array([i, j, 0])
        vertices[:,0] /= (width - 1)
        vertices[:,1] /= (height - 1)

        edges = [[] for _ in range(width * height)]
        faces = []
        c = {}
        for i in range(width - 1):
            for j in range(height - 1):
                v00 = i * height + j
                v01 = i * height + j + 1
                v10 = (i + 1) * height + j
                v11 = (i + 1) * height + j + 1

                edges[v00].append(v11)
                edges[v11].append(v01)
                edges[v01].append(v00)
                faces.append((v00, v11, v01))
                c[v00,v11] = v01
                c[v11,v01] = v00
                c[v01,v00] = v11

                edges[v00].append(v10)
                edges[v10].append(v11)
                edges[v11].append(v00)
                faces.append((v00, v10, v11))
                c[v00,v10] = v11
                c[v10,v11] = v00
                c[v11,v00] = v10

        return vertices, edges, faces, c

    def get_partials(self):
        return self._partials

    def get_vertices(self):
        vertices = np.copy(self._grid)
        vertices[:,2] = self._z
        return vertices

    def get_edges(self):
        return self._edges

    def get_faces(self):
        return self._faces

    def get_boundary_vertices(self):
        boundary_vertices = set()
        boundary_vertices.update(range(self._height))
        boundary_vertices.update(range(self._height * (self._width - 1),
                                       self._height * self._width))
        boundary_vertices.update(range(self._height,
                                       self._height * (self._width - 1),
                                       self._height))
        boundary_vertices.update(range(2 * self._height - 1,
                                       self._height * self._width - 1,
                                       self._height))
        return boundary_vertices

    def get_boundary_edges(self):
        boundary_edges = set()
        for i, j in self._c:
            if (j, i) not in self._c:
                boundary_edges.add((i, j))
        return boundary_edges

    def get_c(self):
        return self._c

    def get_parameters(self):
        return np.copy(self._z)

    def set_parameters(self, z):
        if not np.allclose(self._z, z):
            self._z = np.copy(z)
            self._updates += 1
        return z

    def updates(self):
        return self._updates

    def nearest_vertex_index(self, x, y):
        '''
        Find the index of the vertex whose (x, y) coordinate pair is closest to
        the input coordinate pair. We assume x and y are between 0 and 1,
        inclusive.
        '''

        i = round(x * (self._width - 1))
        j = round(y * (self._height - 1))
        return i * self._height + j

    def coordinates_to_indices(self, coordinates):
        '''
        Convert a list of (x, y) pairs into a list of indices such that the
        coordinates have been approximately scaled and embedded into our mesh.
        '''

        if not coordinates:
            return []

        x_min = coordinates[0][0]
        x_max = coordinates[0][0]
        y_min = coordinates[0][1]
        y_max = coordinates[0][1]

        for x, y in coordinates:
            x_min = min(x_min, x)
            x_max = max(x_max, x)
            y_min = min(y_min, y)
            y_max = max(y_max, y)
        x_divisor = x_max - x_min
        x_divisor = 1. if x_divisor == 0. else x_divisor
        y_divisor = y_max - y_min
        y_divisor = 1. if y_divisor == 0. else y_divisor

        return [self.nearest_vertex_index((x - x_min) / x_divisor,
                                          (y - y_min) / y_divisor)
                for x, y in coordinates]
