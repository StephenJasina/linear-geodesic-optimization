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

def sphere_random(mesh, count=10, connectivity=5):
    # Manually seed for testing purposes
    rng = np.random.default_rng(0)

    v = mesh.get_vertices()
    s_indices = rng.choice(range(v.shape[0]), count, replace=False)
    connections = set()
    for si in s_indices:
        distances = sorted([(np.arccos(np.clip(v[si] @ v[sj]
                                    / np.linalg.norm(v[si])
                                    / np.linalg.norm(v[sj]), -1., 1.)),
                             sj)
                            for sj in s_indices if si != sj])[:connectivity]
        for _, sj in distances:
            connections.add((min(si, sj), max(si, sj)))

    ts = {s_index: [] for s_index in s_indices}
    for si, sj in connections:
        distance = np.arccos(np.clip(v[si] @ v[sj]
                                     / np.linalg.norm(v[si])
                                     / np.linalg.norm(v[sj]), -1., 1.))
        t = distance * max(0.4, rng.normal(1, 0.3))
        ts[si].append((sj, t))
        ts[sj].append((si, t))

    return s_indices, ts
