use_data_path=../../databox/uspto_full/single/database_uspto_full.csv
output_dir=../../databox/uspto_full/single/s2r/
canonical_pd_info_list=../../databox/uspto_full/single/p2s/canonical_pd_info_list

python get_aug_s2r.py -use_data_path $use_data_path \
                      -output_dir $output_dir \
                      -canonical_pd_info_list $canonical_pd_info_list

