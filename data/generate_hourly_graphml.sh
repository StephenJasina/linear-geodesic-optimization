#!/bin/bash

# Very hacky script to call csv_to_graphml.py
for i in $(seq 0 23);
do
	python csv_to_graphml.py -l graph_Europe_hourly/latencies_${i}.csv -p graph_Europe_hourly/probes.csv -o graph_Europe_hourly/graph_${i} -e 7
done
