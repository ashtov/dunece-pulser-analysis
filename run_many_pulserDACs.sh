#!/bin/bash

set -m

#./run_many_pulserDACs.py
for i in {11..26..4}; do
	python find_undershoots.py _config_multirun_$i.tsv INFO 2> fitlogs/log_fitplots_all_$i.log &
done
