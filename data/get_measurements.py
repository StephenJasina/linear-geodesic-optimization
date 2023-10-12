import concurrent.futures
import csv
import datetime
import os
import argparse
from ripe.atlas.cousteau import AtlasResultsRequest
from tqdm import tqdm
#
# def get_measurements(date_start, date_stop,
#                      probes_filename, latencies_filename):
#     rtts = {}
#     def get_rtt(id_measurement, id_source, id_target):
#         while True:
#             is_success, results = AtlasResultsRequest(
#                 msm_id = id_measurement,
#                 start = date_start,
#                 stop = date_stop,
#                 probe_ids = [id_source]
#             ).create()
#
#             if is_success:
#                 latencies = [
#                     result['min']
#                     for result in results
#                     if result['min'] > 0
#                 ]
#                 if not latencies:
#                     rtts[id_source, id_target] = None
#                     print(f'No results for measurement {id_measurement} '
#                           + f'between probes {id_source} and {id_target}')
#                     return None
#
#                 rtt = min(latencies)
#                 rtts[id_source,id_target] = rtt
#                 # print(f'{id_source},{id_target},{rtt}')
#                 return rtt
#
#             if 'MaxRetryError' not in str(results):
#                 print(f'Invalid query for measurement {id_measurement} '
#                       + f'between probes {id_source} and {id_target}',
#                       str(results))
#                 print(date_start, date_stop)
#                 rtts[id_source,id_target] = None
#                 return None
#
#     probes = {}
#     with open(probes_filename, 'r') as probes_file:
#         reader = csv.DictReader(probes_file)
#         probes = [row for row in reader]
#
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         futures = []
#         for probe_target in probes:
#             id_measurement = probe_target['measurement_id']
#             id_target = probe_target['id']
#             for probe_source in probes:
#                 id_source = probe_source['id']
#                 futures.append(
#                     executor.submit(get_rtt,
#                                     id_measurement, id_source, id_target)
#                 )
#
#         for future in futures:
#             future.result()
#
#     with open(latencies_filename, 'w') as latencies_file:
#         writer = csv.DictWriter(
#             latencies_file,
#             ['source_id', 'target_id', 'rtt']
#         )
#         writer.writeheader()
#         for (id_source, id_target), rtt in sorted(rtts.items()):
#             writer.writerow({
#                 'source_id': id_source,
#                 'target_id': id_target,
#                 'rtt': rtt
#             })
#
# if __name__ == '__main__':
#     # ip_type = 'ipv4'
#     # date_start = datetime.datetime(year=2023, month=8, day=29)
#     # hour = datetime.timedelta(hours=1)
#     # for i in range(24):
#     #     get_measurements(date_start + i * hour,
#     #                      date_start + (i + 1) * hour,
#     #                      os.path.join('graph_Europe_hourly', ip_type, f'probes_{ip_type}.csv'),
#     #                      os.path.join('graph_Europe_hourly', ip_type,
#     #                                   f'latencies_{i}.csv'))
#     parser = argparse.ArgumentParser(description='Get measurements for given parameters.')
#
#     parser.add_argument('-i', '--ip_type', type=str, required=True, help='Type of IP (e.g., ipv4, ipv6).')
#     parser.add_argument('-d', '--date', type=str, required=True, help='Start date in YYYY-MM-DD format.')
#     parser.add_argument('-o', '--output', type=str, required=True, help='Output file for latencies.')
#
#     args = parser.parse_args()
#
#     ip_type = args.ip_type
#     date_start_str = args.date
#     output_file = args.output
#
#     date_start = datetime.datetime.strptime(date_start_str, '%Y-%m-%d')
#     hour = datetime.timedelta(hours=1)
#
#     for i in range(1,24):
#         output_hourly_file = os.path.join(output_file,f"latencies_{i}.csv")  # To create hourly files.
#         get_measurements(date_start + i * hour,
#                          date_start + (i + 1) * hour,
#                          os.path.join(ip_type, f'probes_{ip_type}.csv'),
#                          output_hourly_file)

MAX_RETRIES = 3

def get_rtt(id_measurement, id_source, id_target, date_start, date_stop):
    retries = 0
    while retries < MAX_RETRIES:
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
                # print(f"No results for measurement {id_measurement} between probes {id_source} and {id_target}")
                return id_source, id_target, None

        if 'MaxRetryError' not in str(results):
            print(f"Invalid query for measurement {id_measurement} between probes {id_source} and {id_target}")
            return id_source, id_target, None

        retries += 1

    print(f"Max retries reached for measurement {id_measurement} between probes {id_source} and {id_target}")
    return id_source, id_target, None


def get_measurements(date_start, date_stop, probes, latencies_filename):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(get_rtt, probe_target['measurement_id'], probe_source['id'], probe_target['id'], date_start, date_stop): (probe_source['id'], probe_target['id']) for probe_target in probes for probe_source in probes if probe_source['id'] != probe_target['id']}

        rtts = {}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing RTTs"):
            id_source, id_target, rtt = future.result()
            rtts[id_source, id_target] = rtt

    with open(latencies_filename, 'w') as latencies_file:
        writer = csv.DictWriter(latencies_file, ['source_id', 'target_id', 'rtt'])
        writer.writeheader()
        for (id_source, id_target), rtt in rtts.items():
            writer.writerow({
                'source_id': id_source,
                'target_id': id_target,
                'rtt': rtt
            })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get measurements for given parameters.')

    parser.add_argument('-i', '--ip_type', type=str, required=True, help='Type of IP (e.g., ipv4, ipv6).')
    parser.add_argument('-d', '--date', type=str, required=True, help='Start date in YYYY-MM-DD format.')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output file for latencies.')

    args = parser.parse_args()

    ip_type = args.ip_type
    date_start_str = args.date
    output_file = args.output

    # Read probes_filename once and pass it to the function
    with open(os.path.join(ip_type, f'probes_{ip_type}.csv'), 'r') as probes_file:
        reader = csv.DictReader(probes_file)
        probes = [row for row in reader]

    date_start = datetime.datetime.strptime(date_start_str, '%Y-%m-%d')
    hour = datetime.timedelta(hours=1)

    for i in range(17, 24):
        output_hourly_file = os.path.join(output_file, f"latencies_{i}.csv")  # To create hourly files.
        get_measurements(date_start + i * hour, date_start + (i + 1) * hour, probes, output_hourly_file)
