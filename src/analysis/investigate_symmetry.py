import argparse
import csv
import sys

from matplotlib import pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--latencies-file', '-l', type=str, required=True,
                        dest='latencies_filename', metavar='<filename>',
                        help='Input file containing latency information')
    args = parser.parse_args()
    latencies_filename = args.latencies_filename

    with open(latencies_filename, 'r') as f:
        reader = csv.DictReader(f)
        latencies = {}
        for row in reader:
            s = row['source_id']
            d = row['target_id']
            if not row['rtt']:
                continue
            rtt = float(row['rtt'])
            if (s, d) in latencies:
                latencies[(s, d)] = min(latencies[(s, d)], rtt)
            else:
                latencies[(s, d)] = rtt

        differences = []
        for (s, d) in latencies:
            if s >= d:
                continue
            if (d, s) in latencies:
                differences.append(abs(latencies[(s, d)] - latencies[(d, s)]))

        plt.hist(differences, bins=20)
        plt.xlabel('Time Difference (ms)')
        plt.ylabel('Count')
        plt.show()
