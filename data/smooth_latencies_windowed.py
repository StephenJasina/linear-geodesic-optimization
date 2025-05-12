import csv
import enum
import itertools
import os
import pathlib
import sys

import numpy as np

sys.path.append(str(pathlib.PurePath('..', 'src')))
from linear_geodesic_optimization.data import utility


data_directory = pathlib.PurePath('esnet')
probes_file_path = data_directory / 'probes.csv'
links_input_directory = data_directory / 'links'
threshold = 10
window = 2.
links_output_directory = data_directory / 'links_windowed' / f'{threshold}'

def smooth_windowed(zs, window_center, window_lower, window_upper):
    """
    Smooth a single time series of data.

    The idea is that for particular threshold (window_center), we don't
    want the z values to oscillate above and below that value due to
    random instability. So we only allow the time series to cross the
    threshold value when it is sufficiently low (window_lower) or
    sufficiently high (window_upper).
    """

    # Essentially do a 3-state DFA
    State = enum.Enum('State', [('UNKNOWN', 0), ('LOW', 1), ('HIGH', 2)])
    state = State.UNKNOWN

    zs_smooth = []
    for z in zs:
        if z is None:
            # This represents a gap in the data. No smoothing needed
            zs_smooth.append(None)
            state = State.UNKNOWN
        elif state == State.UNKNOWN:
            # If this is the first point in the series (or first point
            # after a gap), just add the point and set the state
            if z <= window_center:
                zs_smooth.append(min(z, window_lower))
                state = State.LOW
            else:
                zs_smooth.append(max(z, window_upper))
                state = State.HIGH
        elif state == State.LOW:
            # If we're on a run of values we believe to be below the
            # threshold, only change states if the value exceeds the
            # upper edge of the window
            if z <= window_upper:
                zs_smooth.append(min(z, window_lower))
            else:
                zs_smooth.append(max(z, window_upper))
                state = State.HIGH
        else:
            # We have state == State.HIGH
            # If we're on a run of values we believe to be above the
            # threshold, only change states if the value is under the
            # lower edge of the window
            if z <= window_lower:
                zs_smooth.append(min(z, window_lower))
                state = State.LOW
            else:
                zs_smooth.append(max(z, window_upper))

    return zs_smooth

# Read the inputs

# For probes, we need a map from probe ids to lat-long pairs (for the
# purpose of computing GCLs)
probes_dict = {}
with open(probes_file_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        probes_dict[row['id']] = (
            float(row['latitude']),
            float(row['longitude'])
        )

links_filenames = list(sorted(os.listdir(links_input_directory)))
links_input_file_paths = [
    links_input_directory / links_filename
    for links_filename in links_filenames
]
edges = set()
headers = set()
links = []
for links_input_file_path in links_input_file_paths:
    print(f'reading {links_input_file_path}')
    with open(links_input_file_path, 'r') as f:
        link_dict = {}
        reader = csv.DictReader(f)
        headers = headers.union(reader.fieldnames)
        for row in reader:
            edge = (row['source_id'], row['target_id'])
            edges.add(edge)
            link_dict[edge] = row
        links.append(link_dict)
edges = list(sorted(edges))
# For debugging purposes, make sure that source_id and target_id are the
# first two columns
headers.remove('source_id')
headers.remove('target_id')
headers = ['source_id', 'target_id'] + list(sorted(headers))

# Smooth
for edge in edges:
    print(f'smoothing {edge}')
    zs = [
        float(link_dict[edge]['rtt']) if edge in link_dict and link_dict[edge]['rtt'] else None
        for link_dict in links
    ]
    gcl = utility.get_GCL(probes_dict[edge[0]], probes_dict[edge[1]])
    zs_smooth = smooth_windowed(
        zs,
        gcl + threshold,
        max(gcl + threshold - window / 2., 0.), # Should not have a negative RTT
        gcl + threshold + window / 2.
    )
    for z_smooth, link_dict in zip(zs_smooth, links):
        if edge in link_dict:
            link_dict[edge]['rtt'] = z_smooth

# Write the outputs
os.makedirs(links_output_directory, exist_ok=True)
latencies_output_file_paths = [
    links_output_directory / latencies_filename
    for latencies_filename in links_filenames
]
for latencies_output_file_path, link_dict in zip(latencies_output_file_paths, links):
    print(f'writing {latencies_output_file_path}')
    with open(latencies_output_file_path, 'w') as f:
        writer = csv.DictWriter(f, headers)
        writer.writeheader()
        for edge in edges:
            writer.writerow(link_dict[edge])
