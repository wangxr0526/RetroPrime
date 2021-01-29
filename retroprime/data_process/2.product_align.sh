
databox=../../databox
data_name=uspto_full
use_data_path=$databox/$data_name/database_all_before_alignment.csv
use_raw_marked_prod_path=$databox/$data_name/rxn_position_info_pd_list_end
output_dir=$databox/$data_name

python alignment_product.py -use_data_path $use_data_path \
      -use_raw_marked_prod_path $use_raw_marked_prod_path \
      -output_dir $output_dir