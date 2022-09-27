'''
This is a testing file used to show that the new implementation of curvature
(and its gradient) match the old implementation found in the surface-gen repo
found at https://github.com/joshuawisc/surface-gen.
'''

import numpy as np

from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh
from linear_geodesic_optimization.optimization import curvature, laplacian

def angle_of_b_in_t(b, T, vertices):
    '''
    for a vertex b which is an index into vertices
    and a triangle T (which is a triple)
    return the angle formed by the two triangles sides at vertex b
    '''
    others = [t for t in T if t != b]
    u = np.array(vertices[others[0]] - vertices[b])
    v = np.array(vertices[others[1]] - vertices[b])
    return np.arccos((u.T @ v) / (np.linalg.norm(u) * np.linalg.norm(v)))

def vertex_curvature(b, t_of_v, vertices):
    '''
    return the curvature at vertex b
    '''
    # find the angles of the edges incident at each vertex b
    # t_of_v[b] is the set of triangles incident to vertex b
    angles = [angle_of_b_in_t(b, t, vertices) for t in t_of_v[b]]
    if len(angles) == 6:
        # this is an interior vertex
        return 2 * np.pi - np.sum(angles)
    else:
        # this vertex does not have the standard number of neighbors (6)
        # which happens at the edge of the mesh
        # so just ignore it - will be filtered out of downstream processing
        return np.nan

def get_curvature(vertices, t_of_v):
    '''
    return an array containing the curvature at each vertex in vertices
    '''
    return np.array([vertex_curvature(v, t_of_v, vertices) for v in range(len(vertices))])

mesh = RectangleMesh(10, 10)
z = np.random.rand(100)
mesh.set_parameters(z)

laplacian_forward = laplacian.Forward(mesh)
laplacian_reverse = laplacian.Reverse(mesh, laplacian_forward)
curvature_forward = curvature.Forward(mesh, [], [], laplacian_forward)
curvature_reverse = curvature.Reverse(mesh, [], [], laplacian_forward, curvature_forward, laplacian_reverse)

curvature_forward.calc()
kappa_0 = curvature_forward.kappa

curvature_reverse.calc(mesh.get_partials()[37], 37)
dif_kappa = curvature_reverse.dif_kappa

# Can't be too much smaller than 1e-5 or we get underflow
delta = 1e-5
z[37] += delta
mesh.set_parameters(z)

curvature_forward.calc()
kappa_delta = curvature_forward.kappa

approx_dif_kappa = np.array([(kappa_delta[i] - kappa_0[i]) / delta for i in range(100)])

# Check curvatures are close
print(np.max(np.abs(np.nan_to_num(get_curvature(mesh.get_vertices(), mesh.triangles_of_vertex()) - curvature_forward.kappa))))

# Check derivative is close
print(np.max(np.abs(approx_dif_kappa - dif_kappa)))
