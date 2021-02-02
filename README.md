# RetroPrime
This is the code for the "RetroPrime: A Chemistry-Inspired and Transformer-based Method for Retro-synthesis Predictions" \
To implement our models we were based on OpenNMT-py (v0.4.1).
# Install requirements
Create a new conda environment:
```
conda install create -n retroprime_env python=3.6
conda activate retroprime_env
conda install -c conda-forge rdkit
conda install pandas tqdm six
conda install pytorch==1.5.0 torchvision cudatoolkit=10.1 -c pytorch
```
Then,
```
cd RetroPrime_root/
pip install -e .
cd RetroPrime_root/retroprime/dataprocess/packeage/SmilesEnumerator/
pip install -e .
```
This step installs the Smiles enumerator. https://github.com/EBjerrum/SMILES-enumeration, or you can also use RDKit's own enumeration function to replace these parts of this code.

Then,
```
cd RetroPrime_root/retroprime/transformer_model/
pip install -e .
```
# Dataset
USPTO-50K: https://github.com/connorcoley/retrosim/blob/master/retrosim/data/data_processed.csv  
USPTO 1976_sep2016: https://figshare.com/articles/dataset/Chemical_reactions_from_US_patents_1976-Sep2016_/5104873
# Data Processing
```
cd RetroPrime_root/
mkdir RetroPrime_root/databox
cd RetroPrime_root/retroprime/dataprocess/
```
You can put the data set (csv) in the path shown below:
```
RetroPrime_root/databox/dataset_name/dataset.csv
```
You can follow the sequence number of the *.sh script. Or you can change the file path in the script to handle your own reaction data. Reaction dataset like this:
```
id,reactants>reagents>production,dataset
US09371281B2,[OH-:1].O[NH2:4].C[O:6][C:7](=O)[c:9]1[cH:10][cH:11][c:12]2[cH:13][cH:14][n:15]([CH2:18][c:19]3[cH:20][cH:21][c:22]([O:25][CH3:26])[cH:23][cH:24]3)[c:16]2[cH:17]1>>[OH:1][NH:4][C:7](=[O:6])[c:9]1[cH:10][cH:11][c:12]2[cH:13][cH:14][n:15]([CH2:18][c:19]3[cH:20][cH:21][c:22]([O:25][CH3:26])[cH:23][cH:24]3)[c:16]2[cH:17]1,train
US07842713B2,[C:26]([CH3:27])([CH3:28])([CH3:29])[O:30][C:31](=[O:32])[c:33]1[cH:34][c:35]2[c:36]([c:43]([OH:45])[cH:44]1)[CH2:37][C:38]([CH3:40])([CH2:41][OH:42])[O:39]2.F[c:55]1[cH:54][cH:53][c:52]([S:49]([CH:46]2[CH2:47][CH2:48]2)(=[O:50])=[O:51])[cH:57][cH:56]1>>[C:26]([CH3:27])([CH3:28])([CH3:29])[O:30][C:31](=[O:32])[c:33]1[cH:34][c:35]2[c:36]([c:43]([O:45][c:55]3[cH:54][cH:53][c:52]([S:49]([CH:46]4[CH2:47][CH2:48]4)(=[O:50])=[O:51])[cH:57][cH:56]3)[cH:44]1)[CH2:37][C:38]([CH3:40])([CH2:41][OH:42])[O:39]2,train
US07642277B2,OOC(c1cccc(Cl)c1)=[O:9].[Cl:12][c:13]1[c:14]([CH2:19][CH:20]=[CH2:21])[cH:15][cH:16][cH:17][cH:18]1>>[O:9]1[CH:20]([CH2:19][c:14]2[c:13]([Cl:12])[cH:18][cH:17][cH:16][cH:15]2)[CH2:21]1,val
US04837349,CC(C)(C)[O:8][C:6]([NH:5][C@H:4]([C:3]([O:2][CH3:1])=[O:16])[CH:13]([CH3:14])[CH3:15])=[O:7].O=S(=O)(O[Si:25]([CH3:26])([CH3:27])[C:28]([CH3:29])([CH3:30])[CH3:31])C(F)(F)F>>[CH3:1][O:2][C:3]([C@@H:4]([NH:5][C:6](=[O:7])[O:8][Si:25]([CH3:26])([CH3:27])[C:28]([CH3:29])([CH3:30])[CH3:31])[CH:13]([CH3:14])[CH3:15])=[O:16],test
```
# Train
Tokenlized data preprocess:
```
dataset=dataset_name_pos_pred
python preprocess.py -train_src data/${dataset}/src-train.txt \
                     -train_tgt data/${dataset}/tgt-train.txt \
                     -valid_src data/${dataset}/src-val.txt \
                     -valid_tgt data/${dataset}/tgt-val.txt \
                     -save_data data/${dataset}/${dataset} \
                     -src_seq_length 1000 -tgt_seq_length 1000 \
                     -src_vocab_size 1000 -tgt_vocab_size 1000 -share_vocab

dataset=dataset_name_S2R
python preprocess.py -train_src data/${dataset}/src-train.txt \
                     -train_tgt data/${dataset}/tgt-train.txt \
                     -valid_src data/${dataset}/src-val.txt \
                     -valid_tgt data/${dataset}/tgt-val.txt \
                     -save_data data/${dataset}/${dataset} \
                     -src_seq_length 1000 -tgt_seq_length 1000 \
                     -src_vocab_size 1000 -tgt_vocab_size 1000 -share_vocab
```
train the two stage model:
```
dataset=dataset_name_pos_pred
python  train.py -data data/${dataset}/${dataset} \
                   -save_model experiments/checkpoints/${dataset}/${dataset}_model \
                   -seed 42 -gpu_ranks 2 -save_checkpoint_steps 10000 -keep_checkpoint 20 \
                   -train_steps 250000 -param_init 0  -param_init_glorot -max_generator_batches 32 \
                   -batch_size 4096 -batch_type tokens -normalization tokens -max_grad_norm 0  -accum_count 4 \
                   -optim adam -adam_beta1 0.9 -adam_beta2 0.998 -decay_method noam -warmup_steps 8000  \
                   -learning_rate 2 -label_smoothing 0.0 -report_every 1000 \
                   -layers 13 -rnn_size 500 -word_vec_size 500 -encoder_type transformer -decoder_type transformer \
                   -dropout 0.1 -position_encoding -share_embeddings \
                   -global_attention general -global_attention_function softmax -self_attn_type scaled-dot \
                   -heads 8 -transformer_ff 2048

dataset=dataset_name_S2R
python  train.py -data data/${dataset}/${dataset} \
                   -save_model experiments/checkpoints/${dataset}/${dataset}_model \
                   -seed 42 -gpu_ranks 2 -save_checkpoint_steps 10000 -keep_checkpoint 20 \
                   -train_steps 250000 -param_init 0  -param_init_glorot -max_generator_batches 32 \
                   -batch_size 4096 -batch_type tokens -normalization tokens -max_grad_norm 0  -accum_count 4 \
                   -optim adam -adam_beta1 0.9 -adam_beta2 0.998 -decay_method noam -warmup_steps 8000  \
                   -learning_rate 2 -label_smoothing 0.0 -report_every 1000 \
                   -layers 13 -rnn_size 500 -word_vec_size 500 -encoder_type transformer -decoder_type transformer \
                   -dropout 0.1 -position_encoding -share_embeddings \
                   -global_attention general -global_attention_function softmax -self_attn_type scaled-dot \
                   -heads 8 -transformer_ff 2048                   
```

#Use model for prediction
We provide a template for reference```run_example.sh``` like this:
```
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
```

You can prepare the input file (in TXT format) and specify the path of the trained two-stage model. Then:
```
./run_example.sh INPUT_NAME OUTPUT_FOLDER BEAM_SIZE
```





