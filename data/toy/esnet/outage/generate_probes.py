import csv
import pathlib
import sys

sys.path.append(str(pathlib.PurePath('..', '..', '..', '..', 'src')))
from linear_geodesic_optimization.data import utility


# Mapping from probes to coordinates
probes = {
    'A': (0, 0),
    'B': (1, 0),
    'C': (1, 1),
    'D': (0, 1),
    'E': (2, 0),
    'F': (3, 0),
    'G': (3, 1),
    'H': (2, 1),
}
latlongs = {}

links = [
    [
        ('A', 'B'),
        ('A', 'C'),
        ('A', 'D'),
        ('B', 'C'),
        ('B', 'D'),
        ('C', 'D'),
        ('E', 'F'),
        ('E', 'G'),
        ('E', 'H'),
        ('F', 'G'),
        ('F', 'H'),
        ('G', 'H'),
        ('B', 'E'),
        ('C', 'H'),
    ],
    [
        ('A', 'B'),
        ('A', 'C'),
        ('A', 'D'),
        ('B', 'C'),
        ('B', 'D'),
        ('C', 'D'),
        ('E', 'F'),
        ('E', 'G'),
        ('E', 'H'),
        ('F', 'G'),
        ('F', 'H'),
        ('G', 'H'),
        ('B', 'E'),
    ],
    [
        ('A', 'B'),
        ('A', 'C'),
        ('A', 'D'),
        ('B', 'C'),
        ('B', 'D'),
        ('C', 'D'),
        ('E', 'F'),
        ('E', 'G'),
        ('E', 'H'),
        ('F', 'G'),
        ('F', 'H'),
        ('G', 'H'),
        ('B', 'E'),
        ('C', 'H'),
    ],
]

with open('probes.csv', 'w') as f:
    writer = csv.DictWriter(f, ['id', 'city', 'country', 'latitude', 'longitude'])
    writer.writeheader()

    x_min = min(x for x, _ in probes.values())
    x_max = max(x for x, _ in probes.values())
    y_min = min(y for _, y in probes.values())
    y_max = max(y for _, y in probes.values())
    scale = max(x_max - x_min, y_max - y_min)
    for id, (x, y) in probes.items():
        longitude, latitude = utility.inverse_mercator((x - x_min) / scale - 0.5, (y - y_min) / scale - 0.5)
        latlongs[id] = (latitude, longitude)
        writer.writerow({
            'id': id,
            'city': id,
            'country': 'Country',
            'latitude': latitude,
            'longitude': longitude,
        })

for i, link_list in enumerate(links):
    with open(pathlib.PurePath('links') / f'{i}.csv', 'w') as f:
        writer = csv.DictWriter(f, ['source_id', 'target_id', 'rtt', 'throughput'])
        writer.writeheader()

        for id_source, id_target in link_list:
            writer.writerow({
                'source_id': id_source,
                'target_id': id_target,
                'rtt': utility.get_GCL(latlongs[id_source], latlongs[id_target]),
                'throughput': 0.2,
            })
