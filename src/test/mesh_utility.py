import meshutility
import numpy as np

verts, faces = meshutility.sphere_cvt(100, iters=100)

u, v = 0, 20
path_edge, path_ratio = meshutility.pygeodesic.find_path(verts, faces, u, v)

# ring_l = [path_edge[0][0]]
# ring_r = [path_edge[0][1]]
# for edge in path_edge[1:]:
#     if ring_l[-1] == edge[0]:
#         ring_r.append(edge[1])
#     elif ring_l[-1] == edge[1]:
#         ring_r.append(edge[0])
#     elif ring_r[-1] == edge[0]:
#         ring_l.append(edge[1])
#     elif ring_r[-1] == edge[1]:
#         ring_l.append(edge[0])
#     else:
#         ring_l.append(edge[0])
#         ring_r.append(edge[1])
# print(path_edge)
# print(ring_l)
# print(ring_r)

pts0 = verts[path_edge[:,0]]
pts1 = verts[path_edge[:,1]]
pts = np.multiply(pts0, 1. - path_ratio[:, np.newaxis]) + \
            np.multiply(pts1, path_ratio[:, np.newaxis])

print(path_edge)
print(path_ratio)
print(pts)
