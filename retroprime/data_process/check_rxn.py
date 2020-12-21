import argparse
import os
import pandas as pd
import torch
from tqdm import tqdm
from retroprime.data_process.utiles import get_split_bond_atom
from multiprocessing import Pool

info_list = ['is_self', 'len1', 'len2', 'len_more', 'no_atom_changed']


def run_task(task):
    index, rxn = task
    check_info = get_split_bond_atom(rxn)
    try:
        if check_info.is_self():
            return index, info_list[0]
        else:
            if len(check_info.get_nei_diff()) == 1:
                return index, info_list[1]
            elif len(check_info.get_nei_diff()) == 2:
                return index, info_list[2]
            elif len(check_info.get_nei_diff()) >= 3:
                return index, info_list[3]
    except:
        return index, info_list[4]


if __name__ == '__main__':
    pool = Pool(8)
    # Designated type of reaction
    opt = argparse.ArgumentParser()
    opt.add_argument('-clean_data', default='../../databox/uspto_full/single/new_raw_all.csv')
    opt.add_argument('-output_dir', default='../../databox/uspto_full/single/index_save/')
    args, _ = opt.parse_known_args()
    database = pd.read_csv(args.clean_data)
    rxn_smiles = database['reactants>reagents>production'].tolist()
    info_dic = {
        'is_self': [],
        'len1': [],
        'len2': [],
        'len_more': [],
        'no_atom_changed': []
    }

    tasks = [(index, rxn) for index, rxn in enumerate(rxn_smiles)]
    for results in tqdm(pool.imap_unordered(run_task, tasks), total=len(tasks)):
        index, info = results
        info_dic[info].append(index)

    for info in info_dic:
        info_dic[info].sort()

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    torch.save(info_dic, os.path.join(output_dir, 'info_dic'))
