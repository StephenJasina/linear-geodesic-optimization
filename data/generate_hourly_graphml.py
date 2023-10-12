import os
import subprocess


def csv_to_graphml(hour, ip_type = 'ipv4', e_value=4):
    base_dir = f"{ip_type}/graph_Europe_hourly"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    latencies_file = os.path.join(base_dir, f"latencies_{hour}.csv")
    output_file = os.path.join(base_dir, str(e_value) ,f"graph_{hour}_{e_value}")

    command = ["python", "csv_to_graphml.py", "-l", latencies_file, "-i", ip_type, "-o", output_file, "-e",
               str(e_value)]
    subprocess.run(command)


def main():
    for i in range(24):
        csv_to_graphml(i, e_value=8)


if __name__ == "__main__":
    main()