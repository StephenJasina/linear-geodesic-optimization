import csv
import itertools
import os

import numpy as np


data_directory = os.path.join('animation_Europe')
latencies_input_directory = os.path.join(data_directory, 'latencies')
latencies_output_directory = os.path.join(data_directory, 'latencies_smooth')

def get_changepoints(z, max_count=None):
    z = np.array(z)
    n = len(z)
    # Compute log-likelihood in O(n^2) time
    dp_sum = np.full((n, n + 1), np.nan)
    dp_sum_square = np.full((n, n + 1), np.nan)
    dp_mean = np.full((n, n + 1), np.nan)
    dp_log_likelihood = np.full((n, n + 1), np.nan)
    for i in range(n):
        dp_sum[i, i] = 0.
        dp_sum_square[i, i] = 0.
        dp_log_likelihood[i, i + 1] = 0.
        for j in range(i + 1, n + 1):
            dp_sum[i, j] = dp_sum[i, j - 1] + z[j - 1]
            dp_sum_square[i, j] = dp_sum_square[i, j - 1] + z[j - 1]**2
            dp_mean[i, j] = dp_sum[i, j] / (j - i)
        for j in range(i + 2, n + 1):
            if dp_sum_square[i, j] == np.inf:
                variance_i_j = np.inf
            else:
                variance_i_j = dp_sum_square[i, j] / (j - i) - dp_mean[i, j]**2
            if variance_i_j <= 0:
                dp_log_likelihood[i, j] = -np.inf
            else:
                dp_log_likelihood[i, j] = -((j - i) / 2) * (np.log(2 * np.pi * variance_i_j) + 1)
    # At this point, dp_log_likelihood[i, j] is the log-likelihood of the
    # MLE of z[i:j].

    dp_bics = []
    dp_changepoints = []
    k_max = max_count if max_count is not None else n
    # Element (l, i) is the best BIC score with l changepoints and i
    # elements (that is, looking at z[:i])
    dp_log_likelihood_partition = np.full((k_max, n + 1), np.nan)
    dp_backtrack = np.full((k_max, n + 1), -1, dtype=int)
    # Fill in the DP table
    for i in range(n + 1):
        dp_log_likelihood_partition[0, i] = dp_log_likelihood[0, i]
    for l in range(1, k_max):
        for i in range(l + 1, n + 1):
            log_likelihoods = dp_log_likelihood_partition[l - 1, l : i] + np.array([
                dp_log_likelihood[i - j, i]
                for j in range(i - l, 0, -1)
            ])
            max_index = np.argmax(log_likelihoods)
            dp_backtrack[l, i] = l + max_index
            dp_log_likelihood_partition[l, i] = log_likelihoods[max_index]

    # Iterate over the number of blocks in the partition. Note that the
    # number of parameters in the model is twice the number of blocks
    k_best = np.argmin([
        2 * k * np.log(n) - 2 * dp_log_likelihood_partition[k - 1, n]  # BIC
        for k in range(1, k_max + 1)
    ]) + 1
    if k_best == 1:
        changepoints = []
        means = [dp_mean[0, -1]]
    else:
        backtrack_path = [dp_backtrack[k_best - 1, n]]
        for l in range(k_best - 2):
            backtrack_path.append(dp_backtrack[k_best - l - 1, backtrack_path[-1]])

        # Backtrack to find the changepoints
        changepoints = list(reversed(backtrack_path))
        means = [dp_mean[0, changepoints[0]]] + [
            dp_mean[changepoints[i], changepoints[i + 1] - 1]
            for i in range(len(changepoints) - 1)
        ] + [dp_mean[changepoints[-1], n]]

    return changepoints, means

def smooth_with_changepoints(z):
    z = list(z)
    if not z:
        return []

    partitions = [
        list(group)
        for _, group in itertools.groupby(
            z, lambda x: x is None or bool(np.isinf(x)) or bool(np.isnan(x))
        )
    ]
    z_smooth = []
    for group in partitions:
        if group[0] is None:
            z_smooth.extend(group)
        else:
            changepoints, means = get_changepoints(group)
            lengths = map(
                lambda x: x[1] - x[0],
                itertools.pairwise([0] + changepoints + [len(group)])
            )
            group_smooth = itertools.chain(
                *[
                    itertools.repeat(mean, length)
                    for length, mean in zip(lengths, means)
                ]
            )
            z_smooth.extend(group_smooth)

    return z_smooth

# Read the inputs
latencies_filenames = list(sorted(os.listdir(latencies_input_directory)))
latencies_input_file_paths = [
    os.path.join(latencies_input_directory, latencies_filename)
    for latencies_filename in latencies_filenames
]
edges = set()
latencies = []
for latencies_input_file_path in latencies_input_file_paths:
    print(f'reading {latencies_input_file_path}')
    with open(latencies_input_file_path, 'r') as f:
        latencies_dict = {}
        reader = csv.DictReader(f)
        for row in reader:
            if not row['rtt']:
                continue
            edge = (row['source_id'], row['target_id'])
            edges.add(edge)
            latencies_dict[edge] = float(row['rtt'])
        latencies.append(latencies_dict)
edges = list(sorted(edges))

# Smooth
for edge in edges:
    print(f'smoothing {edge}')
    z = []
    for latency_dict in latencies:
        z.append(latency_dict[edge] if edge in latency_dict else None)
    for rtt_smoothed, latency_dict in zip(smooth_with_changepoints(z), latencies):
        latency_dict[edge] = rtt_smoothed

# Write the outputs
os.makedirs(latencies_output_directory, exist_ok=True)
latencies_output_file_paths = [
    os.path.join(latencies_output_directory, latencies_filename)
    for latencies_filename in latencies_filenames
]
for latencies_output_file_path, latencies_dict in zip(latencies_output_file_paths, latencies):
    print(f'writing {latencies_output_file_path}')
    with open(latencies_output_file_path, 'w') as f:
        writer = csv.DictWriter(f, ['source_id', 'target_id', 'rtt'])
        writer.writeheader()
        for edge in edges:
            rtt = latencies_dict[edge]
            if rtt is not None:
                writer.writerow({
                    'source_id': edge[0],
                    'target_id': edge[1],
                    'rtt': rtt
                })
