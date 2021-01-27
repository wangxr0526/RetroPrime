import rdkit
import argparse
import re
import numpy as np
import pandas as pd
from rdkit import Chem
import sys
from tqdm import tqdm
from retroprime.data_process.utiles import Execute_grammar_err, get_info_index, c2apbp, split_smiles
from copy import deepcopy


def read_file(file, have_class, write_class=False):
    with open(file, 'r', encoding='utf-8') as f:
        line_list = [line.strip().replace(' ', '') for line in f.readlines()]
        class_mark_list = None
        if have_class:
            line_list_copy = deepcopy(line_list)
            line_list = [re.sub('\<RC_([0-9]+)\>', '', line) for line in line_list_copy]
            if write_class:
                class_mark_list = [re.search('\<RC_([0-9]+)\>', line)[0] for line in line_list_copy]
        else:
            pass
    return line_list, class_mark_list


def pre_list_to_group_list(predict_list, beam_size):
    predict_group_list = []
    one_group_list = []
    for pre in predict_list:
        one_group_list.append(pre)
        if len(one_group_list) == beam_size:
            predict_group_list.append(one_group_list)
            one_group_list = []
    return predict_group_list


def get_info_index(smiles):
    info_index_list = []
    mark = 0
    mol = Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            mark = atom.GetProp('molAtomMapNumber')
            info_index_list.append(atom.GetIdx())
    return list(set(info_index_list)), mark


def mark_canonical_from_mark(canonical_smiles, mark_group):
    '''
    mark_group:([index...],mark)
    '''
    index = mark_group[0]
    mark = mark_group[1]
    mol = Chem.MolFromSmiles(canonical_smiles)
    for atom in mol.GetAtoms():
        if atom.GetIdx() in index:
            atom.SetProp('molAtomMapNumber', mark)
    return Chem.MolToSmiles(mol, canonical=False)


def get_top(rank_list, top):
    count = 0
    for rank in rank_list:
        if rank < top:
            count += 1
        else:
            pass
    return count


def execute_grammar_err(canonical, pre):
    if not Execute_grammar_err(canonical, pre):
        if Chem.MolFromSmiles(pre) is not None:
            mark_group = get_info_index(pre)
            marked_aug_smiles = mark_canonical_from_mark(canonical, mark_group)
            return marked_aug_smiles
        else:
            return ''
    else:
        return ''


def rerank_marked(marked_list, beam_size):
    check = [x for x in marked_list if x is not '']
    rerank_list = check + [''] * (beam_size - len(check))
    assert len(rerank_list) == beam_size
    return rerank_list


def get_mark_apbp(canonical_pos_info_pd):
    # 传入的标记都需要只有一种
    mark = re.findall('\:([0-9]+)\]', canonical_pos_info_pd)[0]
    if mark == '1':
        split = True
        test_mol = Chem.MolFromSmiles(c2apbp(canonical_pos_info_pd))
        if test_mol is None:
            test_mol = Chem.MolFromSmarts(c2apbp(canonical_pos_info_pd))
        rxn_pos_index = []
        for atom in test_mol.GetAtoms():
            if atom.HasProp('molAtomMapNumber'):
                rxn_pos_index.append(atom.GetIdx())
        for atom in test_mol.GetAtoms():
            if atom.GetIdx() in rxn_pos_index:
                atom.SetProp('molAtomMapNumber', '1')
                for nei in atom.GetNeighbors():
                    if nei.GetIdx() not in rxn_pos_index:
                        nei.SetProp('molAtomMapNumber', '2')
        for atom in test_mol.GetAtoms():
            if not atom.HasProp('molAtomMapNumber'):
                atom.SetProp('molAtomMapNumber', '3')
        smiles_apbp_info = Chem.MolToSmiles(test_mol, canonical=False)
        return re.sub('\[\*\:([0-9]+)\]', '', smiles_apbp_info), split
    if mark == '4':
        split = True
        test_mol = Chem.MolFromSmarts(c2apbp(canonical_pos_info_pd))
        rxn_pos_index = []
        for atom in test_mol.GetAtoms():
            if atom.HasProp('molAtomMapNumber'):
                rxn_pos_index.append(atom.GetIdx())
        for atom in test_mol.GetAtoms():
            if atom.GetIdx() in rxn_pos_index:
                atom.SetProp('molAtomMapNumber', '1')
                for nei in atom.GetNeighbors():
                    if nei.GetIdx() not in rxn_pos_index:
                        nei.SetProp('molAtomMapNumber', '2')
        for atom in test_mol.GetAtoms():
            if not atom.HasProp('molAtomMapNumber'):
                atom.SetProp('molAtomMapNumber', '3')
        smiles_apbp_info = Chem.MolToSmiles(test_mol, canonical=True)
        smiles_apbp_info_sub = re.sub('\(\[\*\:([0-9]+)\]\)', '', smiles_apbp_info)
        smiles_apbp_info_sub1 = re.sub('\[\*\:([0-9]+)\]', '', smiles_apbp_info_sub)
        return re.sub('\([=,-]\)', '', smiles_apbp_info_sub1), split
    if mark in ['2', '3']:
        split = False
        test_mol = Chem.MolFromSmiles(canonical_pos_info_pd)
        nei1_pos_index_list = []
        rxn_pos_index_f = []
        for atom in test_mol.GetAtoms():
            if atom.HasProp('molAtomMapNumber'):
                rxn_pos_index_f.append(atom.GetIdx())
                nei_list = []
                for nei in atom.GetNeighbors():
                    nei_list.append(nei.GetIdx())
                if len(nei_list) == 1:
                    nei1_pos_index_list.append(atom.GetIdx())
        if len(nei1_pos_index_list) == 0:
            frist_atom_index = rxn_pos_index_f[0]
        else:
            frist_atom_index = nei1_pos_index_list[0]
        smiles1 = Chem.MolToSmiles(test_mol, rootedAtAtom=frist_atom_index, canonical=True)
        test_mol1 = Chem.MolFromSmiles(smiles1)
        rxn_pos_index = []
        for atom in test_mol1.GetAtoms():
            if atom.HasProp('molAtomMapNumber'):
                rxn_pos_index.append(atom.GetIdx())
        for atom in test_mol1.GetAtoms():
            if atom.GetIdx() in rxn_pos_index:
                atom.SetProp('molAtomMapNumber', '1')
                for nei in atom.GetNeighbors():
                    if nei.GetIdx() not in rxn_pos_index:
                        nei.SetProp('molAtomMapNumber', '2')
        for atom in test_mol1.GetAtoms():
            if not atom.HasProp('molAtomMapNumber'):
                atom.SetProp('molAtomMapNumber', '3')
        smiles_apbp_info1 = Chem.MolToSmiles(test_mol1, canonical=True)
        return smiles_apbp_info1, split


def get_mark_apbp_except_err(smiles):
    if smiles != '': return get_mark_apbp(smiles)[0]
    if smiles == '': return ''


def get_rank(row, base, max_rank):
    for i in range(1, max_rank + 1):
        if get_info_index(row['target']) == get_info_index(row['{}{}'.format(base, i)]):
            return i
    return 0


def smi_tokenizer(smi, regex=False):
    """
    Tokenize a SMILES molecule or reaction
    """
    if regex:
        import re
        pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        tokens = [token for token in regex.findall(smi)]
        assert smi == ''.join(tokens)
        return ' '.join(tokens)
    else:
        return ' '.join(split_smiles(smi))


def main(opt):
    src_file, tgt_file, pre_file, beam_size, have_class, save_rank_resut_file, save_top = opt.src_file, opt.tgt_file, opt.pre_file, opt.beam_size, bool(
        opt.have_class), opt.save_rank_results_file, opt.save_top
    write_to_step2 = bool(opt.write_to_step2)
    write_class = bool(opt.write_class)
    evaluation = bool(opt.evaluation)
    step2_save_file = opt.step2_save_file
    step2_save_top1_file = opt.step2_save_top1_file
    print('have class?', have_class)
    print('Reading files ...')
    # 读取src作为prod canonical smiles参考
    canonical_list, class_mark_list = read_file(src_file, have_class=have_class, write_class=write_class)

    # 读取tgt作为ground truth
    if evaluation:
        ground_true_list, _ = read_file(tgt_file, have_class=have_class)
    else:
        print('Using...')

    # 读取预测结果
    predict_list, _ = read_file(pre_file, have_class=have_class)
    predict_group_list = pre_list_to_group_list(predict_list, beam_size=beam_size)
    if evaluation:
        assert len(canonical_list) == len(ground_true_list) == len(predict_group_list)
    else:
        assert len(canonical_list) == len(predict_group_list)

    total = len(predict_group_list)
    if evaluation:
        if class_mark_list is not None:
            test_df = pd.DataFrame({'index': [i for i in range(len(ground_true_list))],
                                    'target': ground_true_list,
                                    'class_mark': class_mark_list})
        else:
            test_df = pd.DataFrame({'index': [i for i in range(len(ground_true_list))],
                                    'target': ground_true_list})
    else:
        test_df = pd.DataFrame({'index': [i for i in range(len(canonical_list))],
                                'target': ['' for i in range(len(canonical_list))]})

    # rank_dic = {i:0 for i in range(len(predict_group_list))}
    # marked_canoncial_list = []

    list_ = list(zip(canonical_list, predict_group_list))
    for i, (canonical, preds) in tqdm(enumerate(list_)):
        one_marked_list = [execute_grammar_err(canonical, x) for x in preds]
        one_marked_list_rerank = rerank_marked(one_marked_list, beam_size)
        for j in range(beam_size):
            test_df.loc[i, 'marked_prediction_{}'.format(j + 1)] = one_marked_list_rerank[j]

    if evaluation:
        test_df['rank'] = test_df.apply(lambda row: get_rank(row, 'marked_prediction_', beam_size), axis=1)
        # test_df = pd.read_csv(save_rank_resut_file)
        print('computing top results......')
        correct = 0
        top_result_df = pd.DataFrame({'top-N': ['C2C Top-{}'.format(i + 1) for i in range(beam_size)]})
        top_resut_list = []
        for i in range(1, beam_size + 1):
            correct += (test_df['rank'] == i).sum()
            print('C2C Top-{}: {:.1f}%'.format(i, 100 * correct / total))
            top_resut_list.append('{:.1f}%'.format(100 * correct / total))
        top_result_df['step_{}'.format(pre_file.split('.pt')[0].split('_')[-1])] = top_resut_list
        top_result_df.to_csv(save_top, index=False)
        test_df.to_csv(save_rank_resut_file, index=False)
    else:
        test_df.to_csv(save_rank_resut_file, index=False)
    top_num = 3
    if write_to_step2:
        print('Writing step2 src-test.txt......')
        src_test = []
        for index in tqdm(range(total)):
            group_list = []
            for j in range(top_num):
                info_c = test_df['marked_prediction_{}'.format(j + 1)][index]
                try:
                    np.isnan(info_c)
                    group_list.append('')

                except:
                    group_list.append(get_mark_apbp_except_err(info_c))

            group_no_err = [g for g in group_list if g is not '']
            if len(group_no_err) == 0:
                group_list = ['.'] * top_num
            else:
                group_list_copy = deepcopy(group_list)
                group_list = [group_list_copy[0]] * (top_num - len(group_no_err)) + group_no_err

            group_str_list = []
            for apbp in group_list:
                if have_class:
                    str_ = test_df['class_mark'][index] + ' ' + smi_tokenizer(apbp, regex=False)
                else:
                    str_ = smi_tokenizer(apbp, regex=False)
                group_str_list.append(str_)
            src_test.extend(group_str_list)
        with open(step2_save_file, 'w', encoding='utf-8') as f:
            for line in src_test:
                f.write(line + '\n')
        if step2_save_top1_file is not '':
            with open(step2_save_top1_file, 'w', encoding='utf-8') as f:
                for line in src_test[::top_num]:
                    f.write(line + '\n')
        print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='evaluate.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-beam_size', type=int, default=10, help='Beam size')
    parser.add_argument('-src_file', type=str, default='../before2/without_class_all_position_data/src-test.txt')
    parser.add_argument('-tgt_file', type=str, default='../before2/without_class_all_position_data/tgt-test.txt')
    parser.add_argument('-pre_file', type=str,
                        default='../before2/without_class_all_position_data/predictions_USPTO-50K_model_step_90000.pt_on_USPTO-50K_beam10.txt')
    parser.add_argument('-save_rank_results_file', type=str, default='c2c_count.csv')
    parser.add_argument('-save_top', type=str, default='top_results.csv')
    parser.add_argument('-evaluation', type=int, default=1, help='0 is False,1 is True')
    parser.add_argument('-have_class', type=int, default=0, help='0 is False,1 is True')
    parser.add_argument('-write_to_step2', type=int, default=1, help='0 is False,1 is True')
    parser.add_argument('-write_class', type=int, default=1, help='0 is False,1 is True')
    parser.add_argument('-step2_save_file', type=str,
                        default='../before2/without_class_all_position_data/to_step2/src-test.txt')
    parser.add_argument('-step2_save_top1_file', type=str,
                        default='')
    opt = parser.parse_args()
    main(opt)
