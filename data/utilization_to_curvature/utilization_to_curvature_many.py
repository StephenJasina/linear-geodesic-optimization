import os
import pathlib
import subprocess

data_dir = pathlib.PurePath('..', 'Internet2')
probes_path = data_dir / 'probes.csv'
links_dir = data_dir / 'measurements'
output_dir = data_dir / 'graphml_utilization_to_curvature'
os.makedirs(output_dir, exist_ok=True)
for links_filename in sorted(os.listdir(links_dir)):
    links_path = links_dir / links_filename
    name = f'{links_path.stem}.graphml'
    subprocess.run([
        'python',
        'utilization_to_curvature.py',
        '-p', str(probes_path),
        '-l', str(links_path),
        '-o', str(output_dir / name)
    ])
