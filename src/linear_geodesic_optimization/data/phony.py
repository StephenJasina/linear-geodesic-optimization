import numpy as np

from linear_geodesic_optimization.mesh.sphere import Mesh as SphereMesh

def sphere_true(mesh):
    '''
    Take three pairs of (approximately) antipodal points on a sphere mesh and
    prentend that we measured their latencies as the geodesic distances on a
    perfect sphere.

    Latencies are returned as a dictionary sending a mesh index to a list of
    pairs of mesh indices and their corresponding measured latencies
    (essentially, an adjacency list with extra information).
    '''

    lat_long_pairs = [
        (0., 0.),
        (0., 90.),
        (90., 0.),
        (0., 180.),
        (0., -90.),
        (-90., 0.),
    ]
    directions = [SphereMesh.latitude_longitude_to_direction(lat, long)
                  for (lat, long) in lat_long_pairs]
    mesh_indices = {mesh.nearest_direction_index(direction)
                    for direction in directions}
    ts = {mi: [(mj, np.arccos(di @ dj))
               for j, (mj, dj) in enumerate(zip(mesh_indices, directions))
               if (i - j) % 6 != 0]
          for i, (mi, di) in enumerate(zip(mesh_indices, directions))}
    return ts

def sphere_random(mesh, count=10, connectivity=5):
    '''
    Pretend there are `count` many cities on a sphere. Each city will have a
    phony latency measurement to its approximately `connectivity` nearest
    neighbors.

    Latencies are returned as a dictionary sending a mesh index to a list of
    pairs of mesh indices and their corresponding measured latencies
    (essentially, an adjacency list with extra information).
    '''

    # Manually seed for testing purposes (this guarantees reproducibility. This
    # can be removed later if wanted)
    rng = np.random.default_rng(0)

    # Here, we use the trick that on the sphere mesh, the normalized vertices
    # are exactly the partials
    v = mesh.get_partials()

    # First, find the set of pairs of cities for which we have a latency
    # measurement. This avoids having duplicates in our adjacency list.
    connections = set()
    mesh_indices = set(rng.choice(range(v.shape[0]), count, replace=False))
    for mi in mesh_indices:
        # Compute a list of pairs of (true distance, mesh index). Sorting this
        # lets us easily get the closest few indices mj to mi
        distances = [(np.arccos(v[mi] @ v[mj]
                                / np.linalg.norm(v[mi]) / np.linalg.norm(v[mj])),
                      mj)
                     for mj in mesh_indices if mi != mj]
        for _, mj in sorted(distances)[:connectivity]:
            connections.add((min(mi, mj), max(mi, mj)))

    ts = {mesh_index: [] for mesh_index in mesh_indices}
    for mi, mj in connections:
        # Set the measured latency to be some random scaling of the
        # geodesic distance (also ensure that the scaling is positive)
        t = np.arccos(v[mi] @ v[mj]) * max(0.4, rng.normal(1., 0.3))

        # For this dataset, assume that measured latency is symmetric across
        # pairs of cities.
        ts[mi].append((mj, t))
        ts[mj].append((mi, t))

    return ts
