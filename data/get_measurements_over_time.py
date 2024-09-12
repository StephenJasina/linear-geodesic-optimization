import concurrent.futures
import csv
import datetime
import pathlib

from ripe.atlas.cousteau import AtlasResultsRequest, MeasurementRequest, ProbeRequest
from tqdm import tqdm


MAX_RETRIES = 3

def get_rtts(id_source, id_target, time_start, time_end, directory_output):
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
            stop=time_end,
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
            with open(directory_output / f'{id_source}_{id_target}.csv', 'w') as f:
                writer = csv.DictWriter(f, ['time', 'rtt'])
                writer.writeheader()
                for result in results:
                    timestamp = result['timestamp']
                    for latencies_dict in result['result']:
                        rtt = latencies_dict['rtt'] if 'rtt' in latencies_dict else 'timeout'
                        writer.writerow({
                            'time': timestamp,
                            'rtt': rtt
                        })
            return

        if 'MaxRetryError' not in str(results):
            print(f'Invalid query for measurement {id_measurement} between probes {id_source} and {id_target}')
            print(str(results))
            return

    print(f'Max retries reached for measurement {id_measurement} between probes {id_source} and {id_target}')
    return


def write_time_series(probes, directory_output, time_start, time_end):
    with concurrent.futures.ThreadPoolExecutor(100) as executor:
        futures = {
            executor.submit(
                get_rtts,
                id_source, id_target,
                time_start, time_end,
                directory_output
            ): (id_source, id_target)
            for id_target in probes
            for id_source in probes
            if id_source != id_target
        }

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures), desc='Processing RTTs'
        ):
            pass

directory = pathlib.PurePath('animation_Europe')
filename_probes = directory / 'probes.csv'
directory_output = directory / 'time_series'
time_start = datetime.datetime.fromisoformat('2024-08-05T00:00:00')
time_end = datetime.datetime.fromisoformat('2024-08-12T00:00:00')
if __name__ == '__main__':
    with open(filename_probes, 'r') as f:
        reader = csv.DictReader(f)
        probes = [
            row['id']
            for row in reader
        ]

    write_time_series(probes, directory_output, time_start, time_end)
