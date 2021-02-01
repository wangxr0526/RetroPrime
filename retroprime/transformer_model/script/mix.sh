#!/bin/bash
group=$1
core=$2
dataset=USPTO-50K_S2R
model=${dataset}_model_step_100000.pt
data_dir=../experiments/results/${dataset}
beam_size=20
#pre_name=predictions_${model}_beam${beam_size}_top3_${group}.txt
#save_name=predictions_${model}_beam${beam_size}_mix_top3_${group}.txt
pre_name=predictions_${model}_beam${beam_size}test_extract_5w_sec_${group}.txt
save_name=predictions_${model}_beam${beam_size}test_extract_5w_sec_mix_top3_${group}.txt
echo $pre_name
python -u mix_c2c_top3_after_rerank.py -pre_file ${data_dir}/${pre_name} \
          -mix_save_file  ${data_dir}/${save_name} \
          -beam_size=${beam_size} -core $core