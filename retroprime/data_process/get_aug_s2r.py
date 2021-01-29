import os
from itertools import permutations

import pandas as pd
from tqdm import tqdm
import torch
from retroprime.data_process.utiles import get_mark_ab, get_mark_apbp, editdistance, min_distance

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-use_data_path', default='../../databox/select_50k/database_all.csv')
    parser.add_argument('-output_dir', default='../../databox/select_50k/')
    parser.add_argument('-canonical_pd_info_list',
                        default='../../databox/select_50k/canonical_pd_info_list')

    opt = parser.parse_args()

    database = pd.read_csv(opt.use_data_path)
    rxn_smiles = database['reactants>reagents>production']
    prod_smiles = database['prod_smiles']

    data_train = database.loc[database['dataset'] == 'train']
    data_test = database.loc[database['dataset'] == 'test']
    data_val = database.loc[database['dataset'] == 'val']

    train_index = [i for i in data_train.index]
    val_index = [i for i in data_val.index]
    test_index = [i for i in data_test.index]
    set_index_dic = {'train': train_index, 'val': val_index, 'test': test_index}
    print('train:{}\nval:{}\ntest:{}'.format(len(train_index), len(val_index), len(test_index)))

    canonical_pd_info_list = torch.load(opt.canonical_pd_info_list)

    apbp_dic = {}
    for i, marked_prod in tqdm(enumerate(canonical_pd_info_list), total=len(canonical_pd_info_list)):
        apbp_dic[i] = get_mark_apbp(marked_prod)
    torch.save(apbp_dic, os.path.join(opt.output_dir, 'apbp_dic'))

    ab_dic = {}
    for index, rxn in tqdm(enumerate(rxn_smiles), total=len(rxn_smiles)):
        ab_dic[index] = get_mark_ab(rxn)

    torch.save(ab_dic, os.path.join(opt.output_dir, 'ab_dic'))
    distance_min_ab_dic = {}
    for i in tqdm(ab_dic.keys()):
        this_ab_list = ab_dic[i].split('.')
        if len(this_ab_list) == 1:
            distance_min_ab_dic[i] = ab_dic[i]
        elif len(this_ab_list) > 1:
            this_ab_list.reverse()
            ab_reverse = '.'.join(this_ab_list)
            distance_ab = editdistance(ab_dic[i], apbp_dic[i][0])
            distance_ab_reverse = editdistance(ab_reverse, apbp_dic[i][0])
            if distance_ab <= distance_ab_reverse:
                distance_min_ab_dic[i] = ab_dic[i]
            else:
                distance_min_ab_dic[i] = ab_reverse
    torch.save(distance_min_ab_dic, os.path.join(opt.output_dir, 'distance_min_ab_dic'))

    apbp_no_flag_dic = {i: apbp_dic[i][0] for i in range(len(apbp_dic))}
    train_data_list = []
    val_data_list = []
    test_data_list = []
    sorted_index = list(sorted(apbp_no_flag_dic.keys()))
    for index in tqdm(sorted_index):
        if index in train_index:
            train_data_list.append((apbp_no_flag_dic[index], distance_min_ab_dic[index]))
        elif index in val_index:
            val_data_list.append((apbp_no_flag_dic[index], distance_min_ab_dic[index]))
        else:
            test_data_list.append((apbp_no_flag_dic[index], distance_min_ab_dic[index]))
    torch.save(train_data_list, os.path.join(opt.output_dir, 'train_data_list'))
    torch.save(val_data_list, os.path.join(opt.output_dir, 'val_data_list'))
    torch.save(test_data_list, os.path.join(opt.output_dir, 'test_data_list'))

    # aug training data
    aug_train_data_list = []
    for index, (synth, react) in tqdm(enumerate(train_data_list)):
        synth_split = synth.split('.')
        if len(synth_split) == 1:
            aug_train_data_list.append((index, synth, react))
        elif 1 < len(synth_split) <= 3:
            for synth_list in permutations(synth_split, len(synth_split)):
                new_synth = '.'.join(synth_list)
                new_react = min_distance(new_synth, react)
                aug_train_data_list.append((index, new_synth, new_react))
        else:
            aug_train_data_list.append((index, synth, react))

    torch.save(aug_train_data_list, os.path.join(opt.output_dir, 'aug_train_data_list'))

