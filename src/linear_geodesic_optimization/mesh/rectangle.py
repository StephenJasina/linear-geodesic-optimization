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
                vertices[i*width+j:] = np.array([i, j, 0])

        edges = [[] for _ in range(width * height)]
        faces = []
        c = {}
        for i in range(width - 1):
            for j in range(height - 1):
                v00 = i * width + j
                v01 = i * width + j + 1
                v10 = (i + 1) * width + j
                v11 = (i + 1) * width + j + 1

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
        boundary_vertices.update(range(self._width))
        boundary_vertices.update(range(self._width * (self._height - 1),
                                       self._width * self._height))
        boundary_vertices.update(range(self._width,
                                       self._width * (self._height - 1),
                                       self._width))
        boundary_vertices.update(range(2 * self._width - 1,
                                       self._width * self._height - 1,
                                       self._width))
        return boundary_vertices

    def get_boundary_edges(self):
        boundary_edges = set()
        for i, j in enumerate(self._c):
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
