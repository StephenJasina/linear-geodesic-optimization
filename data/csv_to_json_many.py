import os
import pathlib
import subprocess

probes_filepath = pathlib.PurePath('Internet2', 'probes.csv')
links_dir = pathlib.PurePath('Internet2', 'measurements')
output_dir = pathlib.PurePath('Internet2', 'json')
os.makedirs(output_dir, exist_ok=True)
for links_filename in list(sorted(os.listdir(links_dir)))[:25]:
    print(links_filename)
    links_filepath = links_dir / links_filename
    name = f'{links_filepath.stem}.json'
    subprocess.run([
        'python',
        'csv_to_json.py',
        '-p', str(probes_filepath),
        '-l', str(links_filepath),
        '-c', str(500000),
        '-o', str(output_dir / f'{links_filepath.stem}.json')
    ])
