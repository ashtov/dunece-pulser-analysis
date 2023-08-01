#!/bin/bash

set -m

#./run_many_pulserDACs_crp5_high.py
for i in {1..31..2}; do
	python find_undershoots.py cfg_crp5/_config_multirun_crp5_high_$i.tsv INFO 2> fitlogs_crp5/log_high_bonus3_$i.log &
done
