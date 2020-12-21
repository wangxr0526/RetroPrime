import argparse
import os

import torch
from rdkit import Chem
from tqdm import tqdm
import pandas as pd
from retroprime.data_process.utiles import transfor_mark

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-use_data_path',
                        default='../../databox/uspto_full/single/database_uspto_full_before_alignment.csv')
    parser.add_argument('-use_raw_marked_prod_path',
                        default='../../databox/uspto_full/single/rxn_position_info_pd_list_end')
    parser.add_argument('-output_dir',
                        default='../../databox/uspto_full/single/')

    opt = parser.parse_args()
    use_data_path = opt.use_data_path
    use_raw_marked_prod_path = opt.use_raw_marked_prod_path
    database = pd.read_csv(use_data_path)
    prod_smiles = database['prod_smiles'].tolist()
    new_rxn_position_info_pd_list_end = torch.load(use_raw_marked_prod_path)
    assert len(prod_smiles) == len(new_rxn_position_info_pd_list_end)

    err_index_list = []
    canonical_pd_info_list = []
    err = 0
    for index, (marked_prod, prod) in tqdm(enumerate(list(zip(new_rxn_position_info_pd_list_end, prod_smiles))),
                                           total=len(new_rxn_position_info_pd_list_end)):
        try:
            canonical_pd_info_list.append(transfor_mark(marked_prod, prod))
        except:
            err += 1
            canonical_pd_info_list.append(None)
            err_index_list.append(index)
            print('err:', err)
    torch.save(canonical_pd_info_list, os.path.join(opt.output_dir, 'canonical_pd_info_list'))
    new_database = database.drop(err_index_list)
    new_database.to_csv(os.path.join(opt.output_dir, 'database_uspto_full.csv'))
