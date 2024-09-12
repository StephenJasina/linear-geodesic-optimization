import collections
import csv
import datetime
import os
import pathlib


directory_original = pathlib.PurePath('raw_data')
directory_new = pathlib.PurePath('measurements')

links = set()
throughputs = collections.defaultdict(lambda: collections.defaultdict(float))
for filename in os.listdir(directory_original):
    source = filename.split('.')[1].upper()
    description = filename.split(' _ ')[-1]
    description = description[3:description.index('-data')]
    description = description.split('-')
    destination = description[1] if source == description[0] else description[0]
    if source == destination:
        continue
    links.add((source, destination))

    with open(directory_original / filename, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamp = int(row['time']) // 1000
            throughput = row['Output [1h averages]']
            if throughput:
                throughput = float(throughput)
            else:
                throughput = 0.
            throughputs[timestamp][(source, destination)] += throughput

for timestamp, snapshot in throughputs.items():
    filename = datetime.datetime.fromtimestamp(timestamp).strftime('%Y%m%d%H%M%S') + '.csv'
    with open(directory_new / filename, 'w') as f:
        writer = csv.DictWriter(f, ['source_id', 'target_id', 'throughput'])
        writer.writeheader()
        for source, destination in sorted(links):
            throughput = snapshot[source, destination]
            writer.writerow({
                'source_id': source,
                'target_id': destination,
                'throughput': throughput
            })
