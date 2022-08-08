import numpy as np
from linear_geodesic_optimization.mesh.sphere import Mesh as SphereMesh

def sphere_true(mesh):
    lat_long_pairs = [
        (0, 0),
        (0, 90),
        (90, 0),
        (0, 180),
        (0, -90),
        (-90, 0),
    ]
    directions = [SphereMesh.latitude_longitude_to_direction(lat, long)
        for (lat, long) in lat_long_pairs]
    s_indices = [mesh.nearest_direction_index(direction)
        for direction in directions]
    ts = {si: [(sj, np.arccos(dsi @ dsj))
            for j, (sj, dsj) in enumerate(zip(s_indices, directions))
            if (i - j) % 6 != 0]
        for i, (si, dsi) in enumerate(zip(s_indices, directions))}
    return s_indices, ts

def sphere_random(mesh, count=10, connections=20):
    v = mesh.get_vertices()
    V = v.shape[0]
    s_indices = v[np.random.choice(range(v.shape[0]), count)]
    distances = [(np.arccos(v[i] @ v[j] / np.linalg.norm(v[i]) / np.linalg.norm(v[i])), i, j)
        for i in range(V) for j in range(i + 1, V)]
    ts = {}
    for d, i, j in sorted(distances):
        ts[i,j] = d * max(0.4, np.random.normal(1, 0.1))

    return s_indices, ts
