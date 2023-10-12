import concurrent.futures
import csv
import datetime
import os
import argparse
from ripe.atlas.cousteau import AtlasResultsRequest
from tqdm import tqdm

MAX_RETRIES = 3

def get_rtt(id_measurement, id_source, id_target, date_start, date_stop):
    for _ in range(MAX_RETRIES):
        is_success, results = AtlasResultsRequest(
            msm_id=id_measurement,
            start=date_start,
            stop=date_stop,
            probe_ids=[id_source]
        ).create()

        if is_success:
            latencies = [result['min'] for result in results if result['min'] > 0]
            if latencies:
                return id_source, id_target, min(latencies)
            else:
                return id_source, id_target, None

        if 'MaxRetryError' not in str(results):
            print(f'Invalid query for measurement {id_measurement} between probes {id_source} and {id_target}')
            return id_source, id_target, None

    print(f'Max retries reached for measurement {id_measurement} between probes {id_source} and {id_target}')
    return id_source, id_target, None


def get_measurements(probes, latencies_filename, date_start, date_stop):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(
                get_rtt,
                probe_target['measurement_id'],
                probe_source['id'], probe_target['id'],
                date_start, date_stop
            ): (probe_source['id'], probe_target['id'])
            for probe_target in probes
            for probe_source in probes
            if probe_source['id'] != probe_target['id']
        }

        rtts = {}
        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(futures), desc='Processing RTTs'):
            id_source, id_target, rtt = future.result()
            rtts[id_source, id_target] = rtt

    with open(latencies_filename, 'w') as latencies_file:
        writer = csv.DictWriter(latencies_file,
                                ['source_id', 'target_id', 'rtt'])
        writer.writeheader()
        for (id_source, id_target), rtt in rtts.items():
            writer.writerow({
                'source_id': id_source,
                'target_id': id_target,
                'rtt': rtt
            })

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get measurements for given parameters.')

    parser.add_argument('--probes-file', '-p', type=str, required=True,
                        dest='probes_filename', metavar='<filename>',
                        help='Input file containing probes information')
    parser.add_argument('--ip-type', '-i', type=str, required=True,
                        dest='ip_type', metavar='<ipv4/ipv6>',
                        help='Type of IP (e.g., ipv4, ipv6).')
    parser.add_argument('--start', '-s', type=str, required=True,
                        dest='start', metavar='<datetime>',
                        help='Start time in YYYY-mm-ddTHH:MM:SS format.')
    parser.add_argument('--end', '-e', type=str, required=True,
                        dest='end', metavar='<datetime>',
                        help='End time in YYYY-mm-ddTHH:MM:SS format.')
    parser.add_argument('--output', '-o', type=str, required=False,
                        dest='output_filename', metavar='<filename>',
                        help='Output file for latencies.')

    args = parser.parse_args()

    probes_filename = args.probes_filename
    ip_type = args.ip_type
    time_start_str = args.start
    time_end_str = args.end
    output_filename = args.output_filename

    # Read probes_filename once and pass the contents to the function
    with open(probes_filename, 'r') as probes_file:
        reader = csv.DictReader(probes_file)
        probes = [row for row in reader]

    time_start = datetime.datetime.fromisoformat(time_start_str)
    time_end = datetime.datetime.fromisoformat(time_end_str)

    if output_filename is None:
        output_filename = f'{ip_type}/probes_{ip_type}.csv'

    get_measurements(probes, output_filename, time_start, time_end)
