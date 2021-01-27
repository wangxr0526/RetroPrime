#!/usr/bin/env bash

dataset=USPTO-full_S2R
model=${dataset}_model_step_250000.pt

# python score_predictions.py -targets data/${dataset}/tgt-test.txt -beam_size 10 -invalid_smiles \
#                     -predictions experiments/results/predictions_${model}_on_${dataset}_beam10.txt

python score_predictions.py -targets data/${dataset}/tgt-test.txt -beam_size 10 -invalid_smiles \
                    -predictions experiments/results/${dataset}/predictions_${model}_beam10.txt
