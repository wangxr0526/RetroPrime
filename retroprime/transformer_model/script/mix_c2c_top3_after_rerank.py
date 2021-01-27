import argparse
from multiprocessing import Pool

import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')


def clear_map_canonical_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        for atom in mol.GetAtoms():
            if atom.HasProp('molAtomMapNumber'):
                atom.ClearProp('molAtomMapNumber')
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    else:
        return ''


def rerank_group(group, group_size):
    # 去除无效结果
    new_group = [x for x in group if x != '']
    return new_group + ['' for i in range(group_size - len(new_group))]


def mix_top2(list1, list2, rotation=True):
    new_list = []
    assert len(list1) == len(list2)
    if rotation:
        for i, l1 in enumerate(list1[:int(len(list1) // 2)]):
            new_list.append(l1)
            new_list.append(list2[i])
    if not rotation:
        new_list = list1[:int(len(list1) // 2)] + list2[:int(len(list1) // 2)]
    return new_list


def mix_top3(list1, list2, list3, rotation=True, rotation_top3=False):
    # 按照top1:top2:top3=4：4：2的比例混合
    new_list = []
    assert len(list1) == len(list2) == len(list3)
    if rotation:
        if not rotation_top3:
            for i, l1 in enumerate(list1[:int(4 * len(list1) / 10)]):
                new_list.append(l1)
                new_list.append(list2[i])
            for i, l3 in enumerate(list3[:int(2 * len(list1) / 10)]):
                new_list.append(l3)
        if rotation_top3:
            for i, l1 in enumerate(list1[:int(4 * len(list1) / 10)]):
                new_list.append(l1)
                new_list.append(list2[i])
                try:
                    new_list.append(list3[i])
                except:
                    pass

    if not rotation:
        new_list = list1[:int(4 * len(list1) / 10)] + list2[:int(4 * len(list1) / 10)] + list3[
                                                                                         :int(2 * len(list1) / 10)]
    return new_list


def run_tasks(task):
    i, smi = task
    return i, clear_map_canonical_smiles(smi)


def main(opt):

    print('reading prediction...')
    with open(opt.pre_file, 'r', encoding='utf-8') as f:
        prediction = [''.join(x.strip().split(' ')) for x in f.readlines()]


    print('clear map and convert invaild smiles to \'\'')
    pool = Pool(opt.core)
    tasks = [(i, x) for i, x in enumerate(prediction)]
    all_results = []
    for result in tqdm(pool.imap_unordered(run_tasks, tasks), total=len(tasks)):
        all_results.append(result)
    all_results.sort(key=lambda x: x[0])
    prediction_canonical = [x[1] for x in all_results]


    all_results_group_ = np.asanyarray(prediction_canonical).reshape(-1, opt.beam_size).tolist()
    all_results_group = []
    for beam_group in tqdm(all_results_group_):
        all_results_group.append(rerank_group(beam_group, opt.beam_size))


    results_group_for_mix = []
    this_top_mix = []
    for beam_group in all_results_group:
        this_top_mix.append(beam_group)
        if len(this_top_mix) == 3:
            unique_this_top_mix = []
            for group in this_top_mix:
                if group not in unique_this_top_mix:
                    unique_this_top_mix.append(group)
            results_group_for_mix.append(unique_this_top_mix)
            this_top_mix = []

    print('{} group'.format(len(results_group_for_mix)))
    assert len(results_group_for_mix) == len(prediction_canonical) / (3 * opt.beam_size)


    print('mixture results...')
    prediction_single = []
    for this_top_mix in results_group_for_mix:
        if len(this_top_mix) == 1:
            prediction_single.append(this_top_mix[0])
        elif len(this_top_mix) == 2:
            new_group = mix_top2(this_top_mix[0], this_top_mix[1], rotation=True)
            prediction_single.append(new_group)
        elif len(this_top_mix) == 3:
            new_group = mix_top3(this_top_mix[0], this_top_mix[1], this_top_mix[2], rotation=True)
            prediction_single.append(new_group)

    with open(opt.mix_save_file, 'w', encoding='utf-8') as f:
        for group in prediction_single:
            for line in group:
                f.write(line + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='mix_c2c_top3_after_rerank.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-pre_file', type=str,
                        default='')
    parser.add_argument('-mix_save_file', type=str,
                        default='')
    parser.add_argument('-beam_size', type=int,
                        default=20)
    parser.add_argument('-core', type=int,
                        default=8)
    opt = parser.parse_args()
    main(opt)
