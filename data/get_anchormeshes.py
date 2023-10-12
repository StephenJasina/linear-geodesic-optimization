import pandas as pd
import urllib.request
from ripe.atlas.sagan import Result
import json
import os
import datetime
# Global Variables
ANCHOR_DATA = {}
UNKNOWN_SOURCES = []


def fetch_data_from_url(url):
    """Fetches data from the specified URL and returns it as JSON."""
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read().decode("utf-8"))


def infer_anchors(page_num, start_time, i, protocol='IPv4'):
    """Fetch and process anchor measurements from the RIPE Atlas API."""
    date_start = datetime.datetime.strptime(start_time, '%Y-%m-%d')
    hour = datetime.timedelta(hours=1)
    start_time = date_start + i * hour,
    end_time = date_start + (i + 1) * hour
    file_name = f'Datasets/AnchorMeasurements/AnchorMeshes{page_num}.json'
    if os.path.exists(file_name):
        print('File already exists')
        return None  # return a flag indicating no results

    anchor_data = fetch_data_from_url(
        f"https://atlas.ripe.net/api/v2/anchor-measurements/?format=json&page={page_num}")

    if not anchor_data['results']:
        return None  # return a flag indicating no results

    for measurement in anchor_data['results']:
        if measurement['type'] == 'ping' and measurement['is_mesh']:
            target_data = fetch_data_from_url(measurement['target'])
            source = target_data['probe']
            with urllib.request.urlopen(measurement['measurement']) as results:
                for result in tqdm(results.readlines()):
                    measurement_data = json.loads(result.decode("utf-8"))
                    split_result = measurement_data['result'].split('?')
                    timebound_url = split_result[0] + f"?start={start_time}&?end={end_time}format=txt"
                    missing_probes = []
                    missing_indices = []
                    print(timebound_url)

                    # Fetch and process atlas results
                    with urllib.request.urlopen(timebound_url) as atlas_res:
                        if protocol in measurement_data['description']:
                            probe_rtt_min = {}
                            for (i, atlas_result) in tqdm(enumerate(atlas_res.readlines())):
                                atlas_data = Result.get(atlas_result.decode("utf-8"))

                                if atlas_data.rtt_min is not None:
                                    if atlas_data.probe_id in probe_rtt_min.keys():
                                        if probe_rtt_min[atlas_data.probe_id] > atlas_data.rtt_min:
                                            probe_rtt_min[atlas_data.probe_id] = atlas_data.rtt_min
                                    else:
                                        probe_rtt_min.update({atlas_data.probe_id: atlas_data.rtt_min})
                                else:
                                    missing_probes.append(str(atlas_data.probe_id))
                                    missing_indices.append((i))

                            # Check for missing keys in the dictionary
                            failed_count = 0
                            for key in list(set(missing_probes)):
                                try:
                                    print(probe_rtt_min[key])
                                except:
                                    failed_count += 1
                            global_anchor_data[source] = probe_rtt_min
                            break

                        else:
                            break
                        # Save the global dictionary as a JSON file
                        with open('Datasets/AnchorMeasurements/AnchorMeshes' + str(page_num) + '.json',
                                  'w') as outfile:
                            json.dump(global_anchor_data, outfile)
            # The current logic was not clear, so this is a placeholder


def convert_to_latencymatrix(input_path, output_path):
    """Converts measurement data into a latency matrix and saves it as CSV."""

    measurements = {}

    # Read all JSON files in the directory
    for root, _, files in os.walk(input_path):
        for file in files:
            if file.endswith(".json"):
                with open(os.path.join(root, file)) as fp:
                    measurements.update(json.load(fp))

    df = pd.DataFrame(measurements)
    df.to_csv(output_path)
    return df


if __name__ == '__main__':
    page = 1
    start_time =
    for i in range(0,24):
        while True:
            result = infer_anchors(page,i)
            if result is None:  # if no results are returned, break the loop
                break
            page += 1

    # Convert to latency matrix
    # convert_to_latencymatrix('Datasets/AnchorMeasurements/', 'path_to_output.csv')