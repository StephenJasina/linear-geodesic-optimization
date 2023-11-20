import datetime
import os
import subprocess

from generate_hourly_graphml import csv_to_graphml
# This script runs the full pipeline to get from the raw data to the animated manifold.
# The first step is to get the data from RIPE


def full_pipeline_hourly_manifold(ip_type='ipv4', date='2023-08-29', e_value=4,
                                  continent='Europe', directory='graph_Europe_hourly'):
    # first fetch the probes
    # check if the file already exists and if it does, don't run the command
    probes_filename = os.path.join(ip_type, directory, f'probes.csv')
    if not os.path.exists(probes_filename):
        command = ['python', 'get_probes.py', '-i', ip_type, '-c', continent,
                   '-o', probes_filename]
        subprocess.run(command)
    # then get the measurements
    # check if the file already exists and if it does, don't run the command
    if not os.path.exists(os.path.join(ip_type, directory,
                                       'latencies_23.csv')):
        time_initial = datetime.datetime.fromisoformat(date)
        for hour in range(24):
            time_start = time_initial + datetime.timedelta(hours=hour)
            time_end = time_initial + datetime.timedelta(hours=hour+1)
            command = ['python', 'get_measurements.py',
                       '-p', probes_filename,
                       '-i', ip_type,
                       '-s', time_start.isoformat(),
                       '-e', time_end.isoformat(),
                       '-o', os.path.join(ip_type, directory, f'latencies_{hour}.csv')]
            subprocess.run(command)
            csv_to_graphml(hour, ip_type, e_value, directory)

if __name__ == '__main__':
    # full_pipeline_hourly_manifold()
    full_pipeline_hourly_manifold(date='2023-11-08', continent='Australia', directory='graph_Australia_outage')
