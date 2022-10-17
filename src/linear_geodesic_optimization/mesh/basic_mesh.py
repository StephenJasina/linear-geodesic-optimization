class Mesh:
    def __init__(self, vertices, edges, faces,
                 boundary_vertices=set(), boundary_edges=set(), data={}):
        self.vertices = vertices
        self.edges = edges
        self.faces = faces
        self.boundary_vertices = boundary_vertices
        self.boundary_edges = boundary_edges

        # Extra data, like loss function values, etc.
        self.data = data

    def get_vertices(self):
        return self.vertices

    def get_edges(self):
        return self.edges

    def get_faces(self):
        return self.faces

    def get_boundary_vertices(self):
        return self.boundary_vertices

    def get_boundary_edges(self):
        return self.boundary_edges
