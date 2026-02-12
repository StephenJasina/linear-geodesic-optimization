import os
import pathlib
import subprocess

data_dir = pathlib.PurePath('..', 'data', 'Internet2')
json_dir = data_dir / 'json'
output_dir = data_dir / 'images'
os.makedirs(output_dir, exist_ok=True)
for json_filename in sorted(os.listdir(json_dir)):
    json_path = json_dir / json_filename
    name = f'{json_path.stem}.png'
    subprocess.run([
        'python',
        'plot_network.py',
        '-j', str(json_path),
        '-o', str(output_dir / name)
    ])
