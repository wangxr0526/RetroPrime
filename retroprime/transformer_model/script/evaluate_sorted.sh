#!/usr/bin/env bash

dataset=USPTO-50K
beam_size=10
for i in {9..9}
do
model=${dataset}_model_step_${i}0000.pt
echo evaluating ${model} results
nohup python -u evaluate.py -beam_size ${beam_size}\
		      -src_file ../data/${dataset}/src-val_check_sorted.txt \
		      -tgt_file ../data/${dataset}/tgt-val_check_sorted.txt \
		      -pre_file ../experiments/results/predictions_${model}_on_${dataset}_beam${beam_size}_val_sorted.txt \
		      -save_rank_results_file c2c_count_${i}0000.csv -save_top top_results_path/step_${i}0000_top_results_sorted.csv \
		      -have_class 0 -write_to_step2 1 -write_class 0  -step2_save_file to_step2/src-val_top3_step${i}0000.txt -step2_save_top1_file to_step2/src-val_top1_step${i}0000.txt> log/step_${i}0000_sorted.txt
done
