
databox=../../databox/
data_name=select_50k
if [ ! -e $output_dir ];
then
    mkdir -p $output_dir
fi
clean_data=$databox/$data_name/new_raw_all.csv
output_dir=$databox/$data_name/cooked_data
core=32
python get_product_rxn_position_info.py -clean_data $clean_data -output_dir $output_dir -core $core