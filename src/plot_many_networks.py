import os
import subprocess

probes_path = '../data/animation_Europe/probes.csv'
latencies_dir = '../data/animation_Europe/latencies_median/'
for latencies_filename in os.listdir(latencies_dir):
    latencies_path = os.path.join(latencies_dir, latencies_filename)
    subprocess.run([
        'python',
        'plot_network.py',
        '-p', probes_path,
        '-l', latencies_path,
        '-e', '5',
        '-m',
        '-o', os.path.join('networks_median', os.path.splitext(latencies_filename)[0] + '.png')
    ])
