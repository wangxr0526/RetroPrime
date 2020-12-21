use_data_path=../../databox/uspto_full/single/database_uspto_full_before_alignment.csv
use_raw_marked_prod_path=../../databox/uspto_full/single/rxn_position_info_pd_list_end
output_dir=../../databox/uspto_full/single/

python alignment_product.py -use_data_path $use_data_path \
      -use_raw_marked_prod_path $use_raw_marked_prod_path \
      -output_dir $output_dir