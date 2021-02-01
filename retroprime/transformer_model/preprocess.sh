#!/usr/bin/env bash


dataset=select_50k_S2R
python preprocess.py -train_src data/${dataset}/src-train.txt \
                     -train_tgt data/${dataset}/tgt-train.txt \
                     -valid_src data/${dataset}/src-val.txt \
                     -valid_tgt data/${dataset}/tgt-val.txt \
                     -save_data data/${dataset}/${dataset} \
                     -src_seq_length 1000 -tgt_seq_length 1000 \
                     -src_vocab_size 1000 -tgt_vocab_size 1000 -share_vocab
