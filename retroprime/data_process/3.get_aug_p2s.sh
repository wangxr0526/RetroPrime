
databox=../../databox
data_name=select_50k
use_data_path=$databox/$data_name/database_all.csv
output_dir=$databox/$data_name/
canonical_pd_info_list=$databox/$data_name/canonical_pd_info_list

python get_aug_p2s.py -use_data_path $use_data_path \
                  -output_dir $output_dir \
                  -canonical_pd_info_list $canonical_pd_info_list