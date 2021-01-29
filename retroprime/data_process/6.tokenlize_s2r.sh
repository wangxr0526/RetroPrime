databox=../../databox
data_name=select_50k

output_dir=../transformer_model/data/$data_name_S2R
cooked_data_path=$databox/$data_name

python tokenlize_s2r.py -output_dir $output_dir \
                        -cooked_data_path $cooked_data_path