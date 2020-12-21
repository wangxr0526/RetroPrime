import os

import pandas as pd
import torch
from tqdm import tqdm

from retroprime.data_process.utiles import smi_tokenizer


def data2txt(data_dic, set_index_dic, save_path, dataset='train'):
    data_index_sort = sorted(list(data_dic.keys()))
    set_index = set_index_dic[dataset]
    #     src_list = []
    #     tgt_list = []
    src_file = open(save_path + 'src-{}.txt'.format(dataset), 'w', encoding='utf-8')
    tgt_file = open(save_path + 'tgt-{}.txt'.format(dataset), 'w', encoding='utf-8')
    for index in tqdm(data_index_sort):
        if index in set_index:
            this_10_pds = data_dic[index]
            if dataset in ['val', 'test']:
                this_10_pds = this_10_pds[:1]
            else:
                pass
            for j, (prod, marked_prod) in enumerate(this_10_pds):
                src = smi_tokenizer(prod)

                tgt = smi_tokenizer(marked_prod)
                src_file.write(src + '\n')
                tgt_file.write(tgt + '\n')

    #                 src_list.append(src)
    #                 tgt_list.append(tgt)
    #     with open(save_path + 'src_{}.txt'.format(dataset),'w',encoding='utf-8') as f:
    #         for line in src_list:
    #             f.write(line+'\n')
    #     with open(save_path + 'tgt_{}.txt'.format(dataset),'w',encoding='utf-8') as f:
    #         for line in tgt_list:
    #             f.write(line+'\n')
    src_file.close()
    tgt_file.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-use_data_path', default='../../databox/uspto_full/single/database_uspto_full.csv')
    parser.add_argument('-output_dir', default='../transformer_model/data/USPTO-full_pos_pred/')
    parser.add_argument('-uspto_full_pos_pred_aug_10_dic',
                        default='../../databox/uspto_full/single/uspto_full_pos_pred_aug_10_dic')

    opt = parser.parse_args()

    database = pd.read_csv(opt.use_data_path)
    uspto_full_pos_pred_aug_10_dic = torch.load(opt.uspto_full_pos_pred_aug_10_dic)
    data_train = database.loc[database['dataset'] == 'train']
    data_test = database.loc[database['dataset'] == 'test']
    data_val = database.loc[database['dataset'] == 'val']

    train_index = [i for i in data_train.index]
    val_index = [i for i in data_val.index]
    test_index = [i for i in data_test.index]
    set_index_dic = {'train': train_index, 'val': val_index, 'test': test_index}
    print('train:{}\nval:{}\ntest:{}'.format(len(train_index), len(val_index), len(test_index)))
    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)
    for dataset in ['train', 'val', 'test']:
        data2txt(uspto_full_pos_pred_aug_10_dic, set_index_dic, opt.output_dir, dataset)
