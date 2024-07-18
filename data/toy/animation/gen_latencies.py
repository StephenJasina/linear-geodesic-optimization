import csv
import os
import sys

sys.path.append(os.path.join('..', '..', '..', 'src'))
from linear_geodesic_optimization.data import utility



edges = set([
    ('A', 'B'),
    ('A', 'C'),
    ('A', 'E'),
    ('B', 'E'),
    ('B', 'F'),
    ('C', 'E'),
    ('C', 'F'),
    ('E', 'F'),
    ('F', 'H'),
    # ('G', 'H'),
    ('G', 'I'),
    ('G', 'J'),
    ('H', 'I'),
    ('I', 'J'),
])

with open('probes.csv') as f:
    probes = list(csv.DictReader(f))
    for element in probes:
        element['latitude'] = float(element['latitude'])
        element['longitude'] = float(element['longitude'])

with open('latencies.csv', 'w') as f:
    writer = csv.DictWriter(f, ['source_id', 'target_id', 'rtt'])
    writer.writeheader()

    for source in probes:
        source_id = source['id']
        source_lat = source['latitude']
        source_long = source['longitude']
        for target in probes:
            target_id = target['id']
            target_lat = target['latitude']
            target_long = target['longitude']

            latency = utility.get_GCL(
                (source_lat, source_long),
                (target_lat, target_long)
            )

            if source_id != target_id and (source_id, target_id) not in edges and (target_id, source_id) not in edges:
                latency += 100.

            writer.writerow({
                'source_id': source_id,
                'target_id': target_id,
                'rtt': latency,
            })
