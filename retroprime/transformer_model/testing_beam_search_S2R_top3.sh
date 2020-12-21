
#!/usr/bin/env bash

dataset=USPTO-full_S2R
model=${dataset}_model_step_250000.pt
beam_size=10
CUDA_VISIBLE_DEVICES=5 python translate.py -gpu 5 -model experiments/checkpoints/${dataset}/${model} \
                    -src script/to_step2/src-test_top3_step350000.txt \
                    -output experiments/results/${dataset}/predictions_${model}_beam${beam_size}_top3_pipline.txt \
                    -batch_size 64 -replace_unk -max_length 200 -beam_size ${beam_size} -n_best ${beam_size} &
