import collections
import csv
import datetime
import os
import pathlib

# from matplotlib import pyplot as plt
import numpy as np


directory = pathlib.PurePath('animation_Europe')
directory_time_series = directory / 'time_series'
directory_output = directory / 'latencies_test'


def aggregate(bin):
    return np.median(bin)

paths_time_series = os.listdir(directory_time_series)
latencies = collections.defaultdict(dict)
for filename in paths_time_series:
    print(filename)
    # Filenames are in the form `<source>_<target>.csv`
    id_source, id_target = os.path.splitext(filename)[0].split('_')
    with open(directory_time_series / paths_time_series[0], 'r') as f:
        reader = csv.DictReader(f)
        rtts = [row for row in reader]
        time_start = datetime.datetime.fromtimestamp(
            min(
                int(row['time'])
                for row in rtts
            )
        )
        rtts = [
            (
                time_delta.days * 24 + time_delta.seconds // 3600,
                float(row['rtt'])
            )
            for row in rtts
            for time_delta in ((datetime.datetime.fromtimestamp(int(row['time'])) - time_start),)
            if row['rtt'] != 'timeout'
        ]
        bins = collections.defaultdict(list)
        for hour, rtt in rtts:
            bins[hour].append(rtt)
        for hour, bin in bins.items():
            latencies[hour][id_source, id_target] = aggregate(bin)

for hour, rtts in latencies.items():
    with open(directory_output / f'{hour}.csv', 'w') as f:
        writer = csv.DictWriter(f, ['source_id', 'target_id', 'rtt'])
        writer.writeheader()
        for (id_source, id_target), rtt in rtts.items():
            writer.writerow({
                'source_id': id_source,
                'target_id': id_target,
                'rtt': rtt,
            })

# A little bit of testing code to display a time series
# Modify as needed/wanted
# hours = list(sorted(latencies.keys()))
# key = next(iter(latencies[hours[0]]))
# rtts = [latencies[hour][key] for hour in hours]
# plt.plot(hours, rtts)
# plt.show()
