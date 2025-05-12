import time
import os
import pathlib
import subprocess

data_dir = pathlib.PurePath('..', 'data', 'esnet')
epsilon = 6
probes_path = data_dir / 'probes.csv'
links = data_dir / 'links_windowed' / f'{epsilon}'
for latencies_filename in sorted(os.listdir(links)):
    latencies_path = os.path.join(links, latencies_filename)
    name = time.strftime(
        '%Y-%m-%d %H:%M',
        time.localtime(int(os.path.splitext(latencies_filename)[0]) // 1000)
    ) + '.png'
    subprocess.run([
        'python',
        'plot_network.py',
        '-p', str(probes_path),
        '-l', str(latencies_path),
        '-c', '500000',
        '-e', f'{epsilon}',
        '-m',
        '-o', str(data_dir / 'images_windowed' / f'{epsilon}' / name)
    ])
