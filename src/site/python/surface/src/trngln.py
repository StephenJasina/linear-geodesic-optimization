# custom triangulation for plotting, optimization, and regularization

# we compute the triangulation explicitly so as to 
# have complete control over it

import networkx as nx

class trngln:
    
    def __init__(self, gridsize):
        self.gridsize = gridsize
        
    def triangles(self):
        '''
        triangs is a list of 3-tuples
        each 3-tuple consists of the (1D) indices of the triangle's vertices
        '''
        self.triangs = []
        for i in range(self.gridsize-1):
            for j in range(self.gridsize-1):
                self.triangs.append(self.triang1(i,j))
                self.triangs.append(self.triang2(i,j))  
        return self.triangs
    
    def triangles_of_vertex(self):
        '''
        return a dictionary giving the triangles for any vertex
        '''
        try:
            self.triangs
        except NameError:
            self.triangles()
        #
        t_of_v = {}
        for T in self.triangs:
            for pt in T:
                if pt in t_of_v:
                    t_of_v[pt].append(T)
                else:
                    t_of_v[pt] = [T]
        return t_of_v
        
    def coord_to_ndx(self, i, j):
        return i*self.gridsize + j

    def ndx_to_coord(self, b):
        return [(b // self.gridsize), (b % self.gridsize)]

    # each square in the mesh is divided into two triangles
    def triang1(self, i, j):
        return [self.coord_to_ndx(i, j), self.coord_to_ndx(i+1, j),
                self.coord_to_ndx(i, j+1)]

    def triang2(self, i, j):
        return [self.coord_to_ndx(i+1, j), self.coord_to_ndx(i+1, j+1),
                self.coord_to_ndx(i, j+1)]
    
    def mesh_graph(self):
        # create this mesh as a graph as well, for visualization or smoothing
        G = nx.Graph()
        for i in range(self.gridsize):
            for j in range(self.gridsize):
                G.add_node(self.coord_to_ndx(i, j), pos=(i,j))
        #
        for i in range(self.gridsize-1):
            for j in range(self.gridsize-1):
                G.add_edge(self.coord_to_ndx(i, j), self.coord_to_ndx(i+1, j))
                G.add_edge(self.coord_to_ndx(i, j), self.coord_to_ndx(i, j+1))
        #
        for i in range(1, self.gridsize):
            for j in range(self.gridsize-1):
                G.add_edge(self.coord_to_ndx(i, j), self.coord_to_ndx(i-1, j+1))
        #       
        return G
    
    def regularization_graph(self):
        
        G = self.mesh_graph()
        
        # Note: these edges are not in the triangulation
        # However for regularization they are useful to get symmetric regularization
        for i in range(1, self.gridsize):
            for j in range(1, self.gridsize):
                G.add_edge(self.coord_to_ndx(i, j), self.coord_to_ndx(i-1, j-1))
        for j in range(self.gridsize-1):
            G.add_edge(self.coord_to_ndx(self.gridsize-1, j), self.coord_to_ndx(self.gridsize-1, j+1))
        for i in range(self.gridsize-1):
            G.add_edge(self.coord_to_ndx(i, self.gridsize-1), self.coord_to_ndx(i+1, self.gridsize-1))

        return G
