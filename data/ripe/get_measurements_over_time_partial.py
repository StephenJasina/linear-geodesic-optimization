import csv
import datetime
import pathlib

from ripe.atlas.cousteau import AtlasResultsRequest, MeasurementRequest, ProbeRequest


MAX_RETRIES = 3

def get_rtts(id_source, id_target, time_start, time_stop):
    probe_target = next(ProbeRequest(id__in=id_target))
    address_target = probe_target['address_v4']
    id_measurement = None
    for measurement in MeasurementRequest(
        target_ip = address_target,
        is_public = True,
        type = 'ping',
        tags = ['anchoring', 'mesh'],
        status = 2
    ):
        if measurement['af'] == 4 and measurement['description'].startswith(
            'Anchoring Mesh Measurement'
        ):
            id_measurement = measurement['id']
            break

    for _ in range(MAX_RETRIES):
        is_success, results = AtlasResultsRequest(
            msm_id=id_measurement,
            start=time_start,
            stop=time_stop,
            probe_ids=[id_source]
        ).create()

        if is_success:
            latencies = [
                (
                    result['timestamp'],
                    latencies_dict['rtt'] if 'rtt' in latencies_dict else 'timeout'
                )
                for result in results
                for latencies_dict in result['result']
            ]
            return latencies

        if 'MaxRetryError' not in str(results):
            print(f'Invalid query for measurement {id_measurement} between probes {id_source} and {id_target}')
            print(str(results))
            return None

    print(f'Max retries reached for measurement {id_measurement} between probes {id_source} and {id_target}')
    return None

directory = pathlib.PurePath('animation_Europe')
filename_pairs = directory / 'pairs.csv'
directory_output = directory / 'time_series'
time_start = datetime.datetime.fromisoformat('2024-08-05T00:00:00')
time_end = datetime.datetime.fromisoformat('2024-08-12T00:00:00')
if __name__ == '__main__':
    with open(filename_pairs, 'r') as f:
        reader = csv.DictReader(f)
        pairs = [
            (row['source_probe_id'], row['target_probe_id'])
            for row in reader
        ]

    for pair in pairs:
        id_source = pair[0]
        id_target = pair[1]
        rtts = get_rtts(id_source, id_target, time_start, time_end)
        with open(directory_output / f'{id_source}_{id_target}.csv', 'w') as f:
            writer = csv.DictWriter(f, ['time', 'rtt'])
            writer.writeheader()
            for timestamp, rtt in rtts:
                writer.writerow({
                    'time': timestamp,
                    'rtt': rtt
                })
