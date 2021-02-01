#!/usr/bin/env bash

input_file=$1
output_dir=$2
cache_dir=${output_dir}/cache
beam_size=$3
core=8
gpu=0
if [ ! -e ${output_dir} ];
then
    mkdir -p ${output_dir}
fi
if [ ! -e $cache_dir ];
then
    mkdir -p $cache_dir
fi

transformer_root=retroprime/transformer_model
model_save_path=${transformer_root}/experiments/checkpoints
to_stage2_scritp_root=${transformer_root}/script
model_P2S=${model_save_path}/USPTO-50K_pos_pred/USPTO-50K_pos_pred_model_step_90000.pt
model_S2R=${model_save_path}/USPTO-50K_S2R/USPTO-50K_S2R_model_step_100000.pt

python ${to_stage2_scritp_root}/smi_tokenizer.py -input $input_file \
                  -output ${output_dir}/canonical_token_for_input.txt
echo Products to Synthons
CUDA_VISIBLE_DEVICES=${gpu} python ${transformer_root}/translate.py -gpu ${gpu} \
                    -model ${model_P2S} \
                    -src ${output_dir}/canonical_token_for_input.txt \
                    -output ${output_dir}/synthon_predicted.txt \
                    -batch_size 64 -replace_unk -max_length 200 -beam_size ${beam_size} -n_best ${beam_size}
python ${to_stage2_scritp_root}/evaluate.py -beam_size ${beam_size}\
		      -src_file ${output_dir}/canonical_token_for_input.txt \
		      -pre_file ${output_dir}/synthon_predicted.txt \
		      -save_rank_results_file ${cache_dir}/pre_synthons_rank.csv \
		      -save_top ${cache_dir}/pre_synthons_top_results.csv \
		      -write_to_step2  \
		      -core 8 \
		      -step2_save_file ${output_dir}/to_synthon_tokenlized.txt
echo Synthons to Reactants
CUDA_VISIBLE_DEVICES=${gpu} python ${transformer_root}/translate.py -gpu ${gpu} \
                    -model ${model_S2R} \
                    -src ${output_dir}/to_synthon_tokenlized.txt \
                    -output ${output_dir}/reactants_predicted.txt \
                    -batch_size 64 -replace_unk -max_length 200 -beam_size ${beam_size} -n_best ${beam_size}
python ${to_stage2_scritp_root}/mix_c2c_top3_after_rerank.py \
                    -pre_file ${output_dir}/reactants_predicted.txt \
                    -mix_save_file  ${output_dir}/reactants_predicted_mix.txt \
                    -beam_size=${beam_size} -core ${core}
		      
                    






