class Mesh:
    def get_directions(self):
        '''
        Return the (normalized) directions of the mesh. This is useful for
        computing partial derivatives later on. The output of this function
        should be treated as "read only."
        '''

        raise NotImplementedError

    def get_vertices(self):
        '''
        Return the vertices of the mesh.
        '''

        raise NotImplementedError

    def get_edges(self):
        '''
        Return the edges of the mesh adjacency list format. The output of this
        function should be treated as "read only."
        '''

        raise NotImplementedError

    def get_faces(self):
        '''
        Return a list of triples (i, j, k) representing the (indices of the)
        faces of the mesh. Each i -> j -> k is oriented counterclockwise. The
        output of this function should be treated as "read only."
        '''

        raise NotImplementedError

    def get_c(self):
        '''
        Return a map from pairs of indices to indices where (i, j) maps to k
        precisely when i -> j -> k is a face oriented counterclockwise. The
        output of this function should be treated as "read only."
        '''

        raise NotImplementedError

    def get_rho(self):
        '''
        Return the magnitudes of the vertices of this mesh, ordered in the same
        way as the output of `get_vertices`.
        '''

        raise NotImplementedError

    def set_rho(self, rho):
        '''
        Set the magnitudes of the vertices of this mesh, ordered in the same
        way as the output of `get_vertices`.
        '''

        raise NotImplementedError

    def updates(self):
        '''
        Return the number of times `set_rho` has been called. This function is
        an easy (O(1)) way to determine whether the mesh has been updated.
        '''

        raise NotImplementedError
