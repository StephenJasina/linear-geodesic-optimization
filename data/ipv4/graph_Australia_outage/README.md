Data here are generated from

```py get_probes.py -i ipv4 -c Oceania -o ipv4/graph_Australia_outage/probes.csv```

and

```py get_measurements.py -p ipv4/graph_Australia_outage/probes.csv -i ipv4 -s 2023-11-08T00:00:00 -e 2023-11-08T00:01:00 -o ipv4/graph_Australia_outage/latencies.csv```

and similar (repeatedly incrementing the hours of the timestamps).
