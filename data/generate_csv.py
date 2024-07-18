import argparse
import csv
import datetime
import os


from get_probes import get_countries, get_probes, write_probes
from get_measurements import write_measurements

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Get measurements for given parameters.'
    )
    parser.add_argument(
        '--ip-type', '-i', type=str, required=True,
        dest='ip_type', metavar='<ipv4/ipv6>',
        help='Type of IP (e.g., ipv4, ipv6).'
    )
    parser.add_argument(
        '--continent', '-c', type=str, required=False,
        dest='continent', metavar='<continent>',
        help='Continent to which to constrain.'
    )
    parser.add_argument(
        '--start', '-s', type=str, required=True,
        dest='start', metavar='<datetime>',
        help='Start time in YYYY-mm-ddTHH:MM:SS format.'
    )
    parser.add_argument(
        '--end', '-e', type=str, required=True,
        dest='end', metavar='<datetime>',
        help='End time in YYYY-mm-ddTHH:MM:SS format.'
    )
    parser.add_argument(
        '--output', '-o', type=str, required=False,
        dest='output_directory', metavar='<directory>',
        help='The directory in which to store the CSVs.'
    )

    args = parser.parse_args()
    ip_type = args.ip_type
    if args.continent is None:
        countries = {'US': 'United States of America'}
    else:
        countries = get_countries(args.continent)
    time_start = datetime.datetime.fromisoformat(args.start)
    time_end = datetime.datetime.fromisoformat(args.end)
    output_directory = args.output_directory
    os.makedirs(output_directory)

    probes_filename = os.path.join(output_directory, 'probes.csv')
    probes = get_probes(countries, ip_type)
    print(probes)
    with open(probes_filename, 'w') as probes_file:
        write_probes(probes, ip_type, probes_file)

    latency_directory = os.path.join(output_directory, 'latencies')
    os.makedirs(latency_directory)
    time = time_start
    while time < time_end:
        time_string = ''.join(filter(str.isdigit, time.isoformat()))
        time_next = time + datetime.timedelta(hours=1)
        write_measurements(
            probes, os.path.join(latency_directory, time_string + '.csv'),
            time, time_next
        )
        time = time_next
