import pandas as pd
import rdkit
from rdkit import Chem
import os
import sys
import torch
import re
from tqdm import tqdm

import SmilesEnumerator.SmilesEnumerator as se

sme = se.SmilesEnumerator()


def clear_info(smiles):
    mol = Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            atom.ClearProp('molAtomMapNumber')
    return Chem.MolToSmiles(mol, canonical=False)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-use_data_path', default='../../databox/select_50k/database_all.csv')
    parser.add_argument('-output_dir', default='../../databox/select_50k/')
    parser.add_argument('-canonical_pd_info_list', default='../../databox/select_50k/canonical_pd_info_list')

    opt = parser.parse_args()
    database = pd.read_csv(opt.use_data_path)
    prod_smiles = database['prod_smiles'].tolist()
    canonical_pd_info_list = torch.load(opt.canonical_pd_info_list)
    canonical_pd_info_dic = {i: j for i, j in enumerate(canonical_pd_info_list)}
    torch.save(canonical_pd_info_dic, os.path.join(opt.output_dir, 'canonical_pd_info_dic'))

    pos_pred_aug_10_dic = {}
    sorted_index = list(sorted(canonical_pd_info_dic.keys()))
    zip_data = list(zip(sorted_index, prod_smiles))
    for index, prod in tqdm(zip_data):
        # if index % 10000 == 0:
        #     print('index_step:', index)
        #     torch.save(uspto_full_pos_pred_aug_10_dic,
        #                'get_aug_pos_pred/uspto_full_pos_pred_aug_10_dic{}'.format(group))
        marked_can_prod = canonical_pd_info_dic[index]
        this_10_pds = []
        this_10_pds.append((prod, marked_can_prod))
        for j in range(9):
            aug_marked_can_prod = sme.randomize_smiles(marked_can_prod)
            aug_can_prod = clear_info(aug_marked_can_prod)
            this_10_pds.append((aug_can_prod, aug_marked_can_prod))
        pos_pred_aug_10_dic[index] = this_10_pds
    print('done')
    p2s_aug_save_path = os.path.join(opt.output_dir, 'get_aug_pos_pred')
    if not os.path.exists(p2s_aug_save_path):
        os.makedirs(p2s_aug_save_path)
    torch.save(pos_pred_aug_10_dic, os.path.join(p2s_aug_save_path, 'pos_pred_aug_10_dic'))
