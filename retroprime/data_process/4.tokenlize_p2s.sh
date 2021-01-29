databox=../../databox
data_name=select_50k

use_data_path=$databox/$data_name/database_all.csv
output_dir=../transformer_model/data/${data_name}_pos_pred/
pos_pred_aug_10_dic=$databox/$data_name/get_aug_pos_pred/pos_pred_aug_10_dic

python tokenlize_p2s.py -use_data_path $use_data_path \
                        -output_dir $output_dir \
                        -pos_pred_aug_10_dic $pos_pred_aug_10_dic