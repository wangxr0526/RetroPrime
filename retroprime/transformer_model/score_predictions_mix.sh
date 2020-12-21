
#!/usr/bin/env bash

dataset=USPTO-full_S2R
model=${dataset}_model_step_250000.pt
beam_size=10
#python script/mix_c2c_top3_after_rerank.py -pre_file experiments/results/USPTO-full_S2R/predictions_${model}_beam${beam_size}_top3_pipline.txt -beam_size ${beam_size} \
#                 -mix_save_file experiments/results/USPTO-full_S2R/predictions_${model}_beam${beam_size}_top3_pipline_rotation.txt
python score_predictions.py -targets data/${dataset}/tgt-test.txt -beam_size 10 -invalid_smiles \
                    -predictions experiments/results/USPTO-full_S2R/predictions_${model}_beam${beam_size}_top3_pipline_rotation.txt #-save_top all_pipline_step2_top_results.csv
