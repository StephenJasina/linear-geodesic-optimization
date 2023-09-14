import concurrent.futures
import csv
import datetime

from ripe.atlas.cousteau import AtlasResultsRequest

date_start = datetime.datetime(year=2023, month=8, day=29)
date_stop = date_start + datetime.timedelta(days=1)
rtts = {}
def get_rtt(id_measurement, id_source, id_target):
    while True:
        is_success, results = AtlasResultsRequest(
            msm_id = id_measurement,
            start = date_start,
            stop = date_stop,
            probe_ids = [id_source]
        ).create()

        if is_success:
            latencies = [
                result['min']
                for result in results
                if result['min'] > 0
            ]
            if not latencies:
                print(f'No results for measurement {id_measurement} '
                      + f'between probes {id_source} and {id_target}')
                return None

            rtt = min(latencies)
            rtts[id_source,id_target] = rtt
            print(f'{id_source},{id_target},{rtt}')
            return rtt

        if 'MaxRetryError' not in str(results):
            print(f'Invalid query for measurement {id_measurement} '
                  + f'between probes {id_source} and {id_target}')
            return None

probes = {}
with open('probes.csv', 'r') as probes_file:
    reader = csv.DictReader(probes_file)
    probes = [row for row in reader]

with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = []
    for probe_target in probes:
        id_measurement = probe_target['measurement_id']
        id_target = probe_target['id']
        for probe_source in probes:
            id_source = probe_source['id']
            futures.append(
                executor.submit(get_rtt,
                                id_measurement, id_source, id_target)
            )

    for future in futures:
        future.result()

with open('latencies.csv', 'w') as latencies_file:
    writer = csv.DictWriter(
        latencies_file,
        ['source_id', 'target_id', 'rtt']
    )
    writer.writeheader()
    for (id_source, id_target), rtt in sorted(rtts.items()):
        writer.writerow({
            'source_id': id_source,
            'target_id': id_target,
            'rtt': rtt
        })