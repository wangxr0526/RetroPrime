use_data_path=../../databox/uspto_full/single/database_uspto_full.csv
output_dir=../transformer_model/data/USPTO-full_pos_pred/
uspto_full_pos_pred_aug_10_dic=../../databox/uspto_full/single/uspto_full_pos_pred_aug_10_dic

python tokenlize_p2s.py -use_data_path $use_data_path \
                        -output_dir $output_dir \
                        -uspto_full_pos_pred_aug_10_dic $uspto_full_pos_pred_aug_10_dic