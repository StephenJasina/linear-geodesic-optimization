import csv
import itertools
import os
import sys

import numpy as np

sys.path.append(os.path.join('..', 'src'))
from linear_geodesic_optimization.data import utility


data_directory = os.path.join('animation_Europe')
probes_file_path = os.path.join(data_directory, 'probes.csv')
latencies_input_directory = os.path.join(data_directory, 'latencies')
latencies_output_directory = os.path.join(data_directory, 'latencies_windowed_5')
threshold = 5.
window = 5.


def smooth_windowed(z, window_center, window_lower, window_upper):
    z = list(z)
    if not z:
        return []

    z_smooth = []
    z_state = None
    for z_current in z:
        if z_current is None:
            z_smooth.append(None)
        else:
            if z_state is None:
                if z_current <= window_center:
                    z_state = True
                    z_smooth.append(window_lower)
                else:
                    z_state = False
                    z_smooth.append(window_upper)
            elif z_state:
                if z_current < window_upper:
                    z_smooth.append(window_lower)
                else:
                    z_state = False
                    z_smooth.append(window_upper)
            else:
                if z_current <= window_lower:
                    z_state = True
                    z_smooth.append(window_lower)
                else:
                    z_smooth.append(window_upper)

    return z_smooth

# Read the inputs
probes_dict = {}
with open(probes_file_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        probes_dict[row['id']] = (
            float(row['latitude']),
            float(row['longitude'])
        )

latencies_filenames = list(sorted(os.listdir(latencies_input_directory)))
latencies_input_file_paths = [
    os.path.join(latencies_input_directory, latencies_filename)
    for latencies_filename in latencies_filenames
]
edges = set()
latencies = []
for latencies_input_file_path in latencies_input_file_paths:
    print(f'reading {latencies_input_file_path}')
    with open(latencies_input_file_path, 'r') as f:
        latencies_dict = {}
        reader = csv.DictReader(f)
        for row in reader:
            if not row['rtt']:
                continue
            edge = (row['source_id'], row['target_id'])
            if edge[0] not in probes_dict or edge[1] not in probes_dict:
                continue
            edges.add(edge)
            latencies_dict[edge] = float(row['rtt'])
        latencies.append(latencies_dict)
edges = list(sorted(edges))

# Smooth
for edge in edges:
    print(f'smoothing {edge}')
    z = [
        latency_dict[edge] if edge in latency_dict else None
        for latency_dict in latencies
    ]
    gcl = utility.get_GCL(probes_dict[edge[0]], probes_dict[edge[1]])
    z_smooth = smooth_windowed(
        z,
        gcl + threshold,
        max(gcl + threshold - window / 2., 0.),
        gcl + threshold + window / 2.
    )
    for rtt_smoothed, latency_dict in zip(z_smooth, latencies):
        latency_dict[edge] = rtt_smoothed

# Write the outputs
os.makedirs(latencies_output_directory, exist_ok=True)
latencies_output_file_paths = [
    os.path.join(latencies_output_directory, latencies_filename)
    for latencies_filename in latencies_filenames
]
for latencies_output_file_path, latencies_dict in zip(latencies_output_file_paths, latencies):
    print(f'writing {latencies_output_file_path}')
    with open(latencies_output_file_path, 'w') as f:
        writer = csv.DictWriter(f, ['source_id', 'target_id', 'rtt'])
        writer.writeheader()
        for edge in edges:
            rtt = latencies_dict[edge]
            if rtt is not None:
                writer.writerow({
                    'source_id': edge[0],
                    'target_id': edge[1],
                    'rtt': rtt
                })
