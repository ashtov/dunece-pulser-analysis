#!/bin/bash

set -m

./run_many_pulserDACs.py
for i in {11..64..4}; do
	python findpeaks.py _config_multirun_$i.tsv INFO 2> fitlogs/log_fit_$i.log &
done
