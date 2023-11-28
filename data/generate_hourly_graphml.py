import os
import subprocess


def csv_to_graphml(hour, ip_type='ipv4', e_value=4, directory='graph_Europe_hourly'):
    base_dir = os.path.join(ip_type, directory)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    probes_file = os.path.join(base_dir, f'probes.csv')
    latencies_file = os.path.join(base_dir, f'latencies_{hour}.csv')
    output_file = os.path.join(base_dir, str(e_value) ,f'graph_{hour}')

    command = ['python', 'csv_to_graphml.py', '-p', probes_file,
               '-l', latencies_file, '-i', ip_type, '-o', output_file,
               '-e', str(e_value)]
    subprocess.run(command)

def main():
    for i in range(24):
        csv_to_graphml(i, e_value=8)

if __name__ == '__main__':
    main()
