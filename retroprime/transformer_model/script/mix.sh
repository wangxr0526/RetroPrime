#!/bin/bash
group=$1
core=$2
dataset=USPTO-50K_S2R
data_dir=../experiments/results/${dataset}
pre_name=predictions_USPTO-50K_S2R_model_step_100000.pt_beam10test_extract_5w_${group}.txt
save_name=predictions_USPTO-50K_S2R_model_step_100000.pt_beam10test_extract_5w_mix_top3_${group}.txt
python -u mix_c2c_top3_after_rerank.py -pre_file ${data_dir}/${pre_name} \
          -mix_save_file  ${data_dir}/${save_name} \
          -beam_size=10 -core $core