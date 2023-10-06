#!/bin/bash

# Very hacky script to call csv_to_graphml.py
for i in $(seq 0 23);
do
	python ipv4.py -l ipv4/graph_Europe_hourly/latencies_${i}.csv -p ipv4/graph_Europe_hourly/probes.csv -o ipv4/graph_Europe_hourly/graph_${i} -e 7
done
