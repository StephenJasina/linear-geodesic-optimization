import numpy as np

from linear_geodesic_optimization.mesh import mesh

class Mesh(mesh.Mesh):
    '''
    Representation of a mesh that is "approximately" a rectangle. In
    particular, projecting the vertices of the mesh so that their z-coordinates
    are 0 will yield a rectangle. The mesh itself looks like a grid where each
    cell has been cut by its major diagonal.
    '''

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
        grid = np.zeros((width * height, 2))

        # The vertices are ordered lexicographically as (x, y)
        for i in range(width):
            for j in range(height):
                grid[i*height+j:] = np.array([i, j])

        # Normalize the vertices so that the mesh is supported on [0, 1]
        grid[:,0] /= (width - 1)
        grid[:,1] /= (height - 1)

        edges = [[] for _ in range(width * height)]
        faces = []
        c = {}

        # Add edges and faces cell-by-cell
        for i in range(width - 1):
            for j in range(height - 1):
                v00 = i * height + j           # Bottom-left
                v01 = i * height + j + 1       # Top-left
                v10 = (i + 1) * height + j     # Bottom-right
                v11 = (i + 1) * height + j + 1 # Top-right

                # Add the top-left triangle
                edges[v00].append(v11)
                edges[v11].append(v01)
                edges[v01].append(v00)
                faces.append((v00, v11, v01))
                c[v00,v11] = v01
                c[v11,v01] = v00
                c[v01,v00] = v11

                # Add the bottom-right triangle
                edges[v00].append(v10)
                edges[v10].append(v11)
                edges[v11].append(v00)
                faces.append((v00, v10, v11))
                c[v00,v10] = v11
                c[v10,v11] = v00
                c[v11,v00] = v10

        return grid, edges, faces, c

    def get_partials(self):
        return self._partials

    def get_vertices(self):
        return np.concatenate((self._grid,
                               np.reshape(self._z, (-1, 1))), axis=1)

    def get_edges(self):
        return self._edges

    def get_faces(self):
        return self._faces

    def get_boundary_vertices(self):
        boundary_vertices = set()

        # Left edge
        boundary_vertices.update(range(self._height))

        # Right edge
        boundary_vertices.update(range(self._height * (self._width - 1),
                                       self._height * self._width))

        # Bottom edge
        boundary_vertices.update(range(self._height,
                                       self._height * (self._width - 1),
                                       self._height))

        # Top edge
        boundary_vertices.update(range(2 * self._height - 1,
                                       self._height * self._width - 1,
                                       self._height))

        return boundary_vertices

    def get_boundary_edges(self):
        # Boundary edges are those that appear as a half-edge in one direction,
        # but not in the opposite direction.
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

        # Need this check to avoid out-of-bounds errors if coordinates is empty
        if not coordinates:
            return []

        x_min = coordinates[0][0]
        x_max = coordinates[0][0]
        y_min = coordinates[0][1]
        y_max = coordinates[0][1]

        # Find scaling factors so that the x range and y range are both [0, 1].
        # If no such scaling factor exists (i.e., the x-coordinate or
        # y-coordinate is constant), set it to some arbitrary value (in this
        # case, just set it to 1.)
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
