import csv
import datetime

from ripe.atlas.cousteau import AtlasResultsRequest

date_start = datetime.datetime(year=2023, month=8, day=29)
date_end = date_start + datetime.timedelta(days=1)

probes = {}
with open('probes.csv', 'r') as probes_file:
    reader = csv.DictReader(probes_file)
    probes = [row for row in reader]

rtts = {}
for probe_destination in probes:
    measurement_id = probe_destination['measurement_id']
    id_destination = probe_destination['id']
    for probe_source in probes:
        id_source = probe_source['id']
        if id_source == id_destination:
            continue
        is_success, results = AtlasResultsRequest(
            msm_id = measurement_id,
            start = date_start,
            end = date_end,
            probe_ids = [id_source]
        ).create()

        if not is_success:
            print(f'Invalid query for measurement {measurement_id} '
                  + f'between probes {id_source} and {id_destination}')
            continue

        latencies = [
            result['min']
            for result in results
            if result['min'] > 0
        ]
        if not latencies:
            print(f'No results for measurement {measurement_id} '
                  + f'between probes {id_source} and {id_destination}')
            continue

        rtt = min(latencies)
        rtts[id_source,id_destination] = rtt
        print(f'{id_source},{id_destination},{rtt}')

with open('latencies.csv', 'w') as latencies_file:
    writer = csv.DictWriter(
        latencies_file,
        ['source_id', 'destination_id', 'rtt']
    )
    writer.writeheader()
    for (id_source, id_destination), rtt in sorted(rtts.items()):
        writer.writerow({
            'source_id': id_source,
            'destination_id': id_destination,
            'rtt': rtt
        })
