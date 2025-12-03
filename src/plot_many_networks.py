import time
import os
import pathlib
import subprocess

data_dir = pathlib.PurePath('..', 'data', 'toy', 'routing_with_volumes')
json_dir = data_dir / 'graphs'
for json_filename in sorted(os.listdir(json_dir)):
    json_path = json_dir / json_filename
    name = f'{json_path.stem}.png'
    subprocess.run([
        'python',
        'plot_network.py',
        '-j', str(json_path),
        '-o', str(data_dir / 'images' / name)
    ])
