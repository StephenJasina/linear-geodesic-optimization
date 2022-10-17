import csv
import os

from linear_geodesic_optimization.mesh.sphere import Mesh as SphereMesh

def sphere_north_america(mesh):
    '''
    Project the North America data onto a sphere mesh.

    Latencies are returned as a dictionary sending a mesh index to a list of
    pairs of mesh indices and their corresponding measured latencies
    (essentially, an adjacency list with extra information).
    '''

    ts = {}

    # Map from city codes (arbitrary integers) to the nearest corresponding
    # vertices in the mesh
    city_id_to_mesh_index = {}

    with open(os.path.join('linear_geodesic_optimization', 'data',
                           'north_america',
                           'locations.csv')) as locations_file:
        locations_reader = csv.reader(locations_file)

        # Skip the header
        next(locations_reader)

        for row in locations_reader:
            city_id = int(row[0])
            latitude = float(row[1])
            longitude = float(row[2])
            direction = SphereMesh.latitude_longitude_to_direction(latitude,
                                                                   longitude)
            mesh_index = mesh.nearest_direction_index(direction)
            city_id_to_mesh_index[city_id] = mesh_index
            ts[mesh_index] = []

    with open(os.path.join('linear_geodesic_optimization', 'data',
                           'north_america',
                           'latencies.csv')) as latencies_file:
        latencies_reader = csv.reader(latencies_file)

        # Skip the header
        next(latencies_reader)

        for row in latencies_reader:
            if row[2] != '':
                mesh_index_a = city_id_to_mesh_index[int(row[0])]
                mesh_index_b = city_id_to_mesh_index[int(row[1])]
                latency = float(row[2])
                ts[mesh_index_a].append((mesh_index_b, latency))

    return ts

def rectangle_north_america(mesh):
    '''
    Project the North America data onto a rectangle mesh.

    Latencies are returned as a dictionary sending a mesh index to a list of
    pairs of mesh indices and their corresponding measured latencies
    (essentially, an adjacency list with extra information).
    '''

    # Parallel lists of city codes (arbitrary integers) with their
    # corresponding (latitude, longitude) coordinates
    city_ids = []
    coordinates = []

    with open(os.path.join('linear_geodesic_optimization', 'data',
                           'north_america',
                           'locations.csv')) as locations_file:
        locations_reader = csv.reader(locations_file)

        # Skip the header
        next(locations_reader)

        for row in locations_reader:
            city_id = int(row[0])
            latitude = float(row[1])
            longitude = float(row[2])

            city_ids.append(city_id)
            coordinates.append((latitude, longitude))
    mesh_indices = [mesh.nearest_vertex_index(x, y)
                    for x, y in mesh.scale_coordinates_to_unit_square(coordinates)]

    # Map from city codes (arbitrary integers) to the nearest corresponding
    # vertices in the mesh
    city_id_to_mesh_index = {city_id: mesh_index
                             for city_id, mesh_index in zip(city_ids,
                                                            mesh_indices)}

    ts = {mesh_index: [] for mesh_index in mesh_indices}

    with open(os.path.join('linear_geodesic_optimization', 'data',
                           'north_america',
                           'latencies.csv')) as latencies_file:
        latencies_reader = csv.reader(latencies_file)

        # Skip the header
        next(latencies_reader)

        for row in latencies_reader:
            if row[2] != '':
                mesh_index_a = city_id_to_mesh_index[int(row[0])]
                mesh_index_b = city_id_to_mesh_index[int(row[1])]
                latency = float(row[2])
                ts[mesh_index_a].append((mesh_index_b, latency))

    return ts
