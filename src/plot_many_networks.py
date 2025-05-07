import time
import os
import pathlib
import subprocess

esnet_dir = pathlib.PurePath('/', 'home', 'jasina', 'research', 'esnet')
probes_path = esnet_dir / 'data' / 'probes.csv'
links = esnet_dir / 'data' / 'links'
epsilon = 7
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
        '-o', str(esnet_dir / 'images' / 'networks' / f'{epsilon}' / name)
    ])
