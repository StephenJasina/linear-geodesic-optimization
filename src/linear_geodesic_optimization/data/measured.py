import csv
import os

from linear_geodesic_optimization.mesh.sphere import Mesh as SphereMesh

def sphere_north_america(mesh):
    city_id_to_index = {}
    s_indices = []
    with open(os.path.join('linear_geodesic_optimization', 'data',
                           'locations.csv')) as locations_file:
        locations_reader = csv.reader(locations_file)
        next(locations_reader)
        for row in locations_reader:
            city_id = int(row[0])
            latitude = float(row[1])
            longitude = float(row[2])
            direction = SphereMesh.latitude_longitude_to_direction(latitude,
                                                                   longitude)
            index = mesh.nearest_direction_index(direction)
            city_id_to_index[city_id] = index
            s_indices.append(index)

    ts = {s_index: [] for s_index in s_indices}
    with open(os.path.join('linear_geodesic_optimization', 'data',
                           'latencies.csv')) as locations_file:
        latencies_reader = csv.reader(locations_file)
        next(latencies_reader)
        for row in latencies_reader:
            index_a = city_id_to_index[int(row[0])]
            index_b = city_id_to_index[int(row[1])]
            if row[2] != '':
                latency = float(row[2])
                ts[index_a].append((index_b, latency))

    return s_indices, ts

def rectangle_north_america(mesh):
    city_ids = []
    coordinates = []
    with open(os.path.join('linear_geodesic_optimization', 'data',
                           'locations.csv')) as locations_file:
        locations_reader = csv.reader(locations_file)
        next(locations_reader)
        for row in locations_reader:
            city_id = int(row[0])
            latitude = float(row[1])
            longitude = float(row[2])

            city_ids.append(city_id)
            coordinates.append((latitude, longitude))
    s_indices = mesh.coordinates_to_indices(coordinates)
    city_id_to_index = {city_id: index
                        for city_id, index in zip(city_ids, s_indices)}

    ts = {s_index: [] for s_index in s_indices}
    with open(os.path.join('linear_geodesic_optimization', 'data',
                           'latencies.csv')) as locations_file:
        latencies_reader = csv.reader(locations_file)
        next(latencies_reader)
        for row in latencies_reader:
            index_a = city_id_to_index[int(row[0])]
            index_b = city_id_to_index[int(row[1])]
            if row[2] != '':
                latency = float(row[2])
                ts[index_a].append((index_b, latency))

    return s_indices, ts
