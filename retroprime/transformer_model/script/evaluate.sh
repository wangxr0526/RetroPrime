#!/usr/bin/env bash

dataset=USPTO-50K
beam_size=10
for i in {16..20}
do
model=${dataset}_model_step_${i}0000.pt
echo evaluating ${model} results
nohup python -u evaluate.py -beam_size ${beam_size}\
		      -src_file ../data/${dataset}/src-val_check.txt \
		      -tgt_file ../data/${dataset}/tgt-val_check.txt \
		      -pre_file ../experiments/results/predictions_${model}_on_${dataset}_beam${beam_size}_val.txt \
		      -save_rank_results_file c2c_count_${i}0000.csv -save_top top_results_path/step_${i}0000_top_results.csv> log/step_${i}0000.txt
done
