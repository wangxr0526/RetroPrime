output_dir=../transformer_model/data/USPTO-full_S2R/
cooked_data_path=../../databox/uspto_full/single/s2r/

python tokenlize_s2r.py -output_dir $output_dir \
                        -cooked_data_path $cooked_data_path