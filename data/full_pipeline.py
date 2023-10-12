from generate_hourly_graphml import csv_to_graphml
import os
import subprocess
# This script runs the full pipeline to get from the raw data to the animated manifold.
# The first step is to get the data from RIPE


def full_pipeline_hourly_manifold(ip_type='ipv4', date='2023-08-29', e_value=4,
                                  directory='graph_Europe_hourly'):
    # first fetch the probes
    # check if the file already exists and if it does, don't run the command
    probes_filename = os.path.join(ip_type, directory, f'probes_{ip_type}.csv')
    if not os.path.exists(probes_filename):
        command = ['python', 'get_probes.py', '-i', 'ipv4', '-c', 'Europe',
                   '-o', probes_filename]
        subprocess.run(command)
    # then get the measurements
    # check if the file already exists and if it does, don't run the command
    if not os.path.exists(os.path.join(ip_type, directory,
                                       'latencies_23.csv')):
        for hour in range(24):
            command = ['python', 'get_measurements.py', '-i', ip_type,
                       '-s', f'{date}T{hour:02}:00:00',
                       '-e', f'{date}T{hour+1:02}:00:00',
                       '-o', os.path.join(ip_type, directory)]
            subprocess.run(command)
            csv_to_graphml(hour, ip_type, e_value, directory)

if __name__ == '__main__':
    full_pipeline_hourly_manifold()
