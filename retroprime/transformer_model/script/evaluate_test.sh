#!/usr/bin/env bash

dataset=USPTO-full_pos_pred
beam_size=10
for i in {35..35}
do
model=${dataset}_model_step_${i}0000.pt
echo evaluating ${model} results
nohup python -u evaluate.py -beam_size ${beam_size}\
		      -src_file ../data/${dataset}/src-test.txt \
		      -tgt_file ../data/${dataset}/tgt-test.txt \
		      -pre_file ../experiments/results/${dataset}/predictions_${model}_beam${beam_size}.txt \
		      -save_rank_results_file c2c_count_${i}0000_test.csv -save_top top_results_path/step_${i}0000_top_results_test_sorted.csv\
		      -have_class 0 -write_to_step2 1 -write_class 0  -step2_save_file to_step2/src-test_top3_step${i}0000.txt -step2_save_top1_file to_step2/src-test_top1_step${i}0000.txt> log/step_${i}0000_sorted.txt
done
