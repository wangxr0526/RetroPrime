#!/usr/bin/env bash

dataset=USPTO-50K
model=${dataset}_model_average.pt

python translate.py -model experiments/models/${model} \
                    -src data/${dataset}/src-test.txt \
                    -output experiments/results/predictions_${model}_on_${dataset}_test.txt \
                    -batch_size 64 -replace_unk -max_length 200 -fast