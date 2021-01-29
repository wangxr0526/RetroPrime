import os
import pandas as pd
import torch
from tqdm import tqdm

from retroprime.data_process.utiles import smi_tokenizer


def datals2txt_S2R(data_list, save_path, dataset='val'):
    src_file = open(save_path + 'src-{}.txt'.format(dataset), 'w', encoding='utf-8')
    tgt_file = open(save_path + 'tgt-{}.txt'.format(dataset), 'w', encoding='utf-8')
    for synth, react in tqdm(data_list):
        src = smi_tokenizer(synth)
        tgt = smi_tokenizer(react)
        src_file.write(src + '\n')
        tgt_file.write(tgt + '\n')
    src_file.close()
    tgt_file.close()


def datals2txt_S2R_aug_train(data_list, save_path):
    src_file = open(save_path + 'src-train.txt', 'w', encoding='utf-8')
    tgt_file = open(save_path + 'tgt-train.txt', 'w', encoding='utf-8')
    for index, synth, react in tqdm(data_list):
        src = smi_tokenizer(synth)
        tgt = smi_tokenizer(react)
        src_file.write(src + '\n')
        tgt_file.write(tgt + '\n')
    src_file.close()
    tgt_file.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-output_dir', default='../transformer_model/data/select_50k_S2R/')
    parser.add_argument('-cooked_data_path',
                        default='../../databox/select_50k/')

    opt = parser.parse_args()
    output_dir = opt.output_dir
    cooked_data_path = opt.cooked_data_path
    cooked_data_name_list = ['aug_train_data_list', 'val_data_list', 'test_data_list']

    cooked_data_dic = {
        'train': torch.load(os.path.join(cooked_data_path, cooked_data_name_list[0])),
        'val': torch.load(os.path.join(cooked_data_path, cooked_data_name_list[1])),
        'test': torch.load(os.path.join(cooked_data_path, cooked_data_name_list[2]))
    }

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    datals2txt_S2R_aug_train(cooked_data_dic['train'], output_dir)
    datals2txt_S2R(cooked_data_dic['val'], output_dir, 'val')
    datals2txt_S2R(cooked_data_dic['test'], output_dir, 'test')
