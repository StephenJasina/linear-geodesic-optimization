from generate_hourly_graphml import csv_to_graphml
import os
import subprocess
# This script runs the full pipeline to get from the raw data to the animated manifold.
# The first step is to get the data from RIPE


def full_pipeline_hourly_manifold(ip_type='ipv4', date = '2023-08-29', e_value=4):
    # first fetch the probes
    # check if the file already exists and if it does, don't run the command
    if not os.path.exists(f"{ip_type}/probes_{ip_type}.csv"):
        command = ["python", "get_probes.py", "-i", "ipv4"]
        subprocess.run(command)
    # then get the measurements
    #check if the file already exists and if it does, don't run the command
    if not os.path.exists(f"{ip_type}/graph_Europe_hourly/latencies_23.csv"):
        command = ["python", "get_measurements.py", "-i", ip_type, "-d", date, "-o",
                   f"{ip_type}/graph_Europe_hourly/"]
        subprocess.run(command)
    for i in range(24):
        csv_to_graphml(i, ip_type, e_value)

if __name__ == "__main__":
    full_pipeline_hourly_manifold()