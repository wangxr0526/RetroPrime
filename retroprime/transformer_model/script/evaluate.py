import warnings
from multiprocessing import Pool

warnings.filterwarnings("ignore")
import argparse
import numpy as np
import pandas as pd
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')
from tqdm import tqdm
from retroprime.data_process.utiles import split_smiles, \
    execute_grammar_err, rerank_marked, read_file, pre_list_to_group_list, get_rank, get_mark_apbp_except_err
from copy import deepcopy


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


def run_tasks(task):
    i, canonical, preds = task
    beam_size = opt.beam_size
    one_marked_list = [execute_grammar_err(canonical, x) for x in preds]
    one_marked_list_rerank = rerank_marked(one_marked_list, beam_size)
    return i, one_marked_list_rerank


def main(opt):
    src_file, tgt_file, pre_file, beam_size, have_class, save_rank_resut_file, save_top = \
        opt.src_file, opt.tgt_file, opt.pre_file, opt.beam_size, bool(opt.have_class), \
        opt.save_rank_results_file, opt.save_top
    write_to_step2 = opt.write_to_step2
    write_class = opt.write_class
    evaluation = opt.evaluation
    step2_save_file = opt.step2_save_file
    step2_save_top1_file = opt.step2_save_top1_file
    print('have reaction type?', have_class)
    print('Reading files ...')
    # 读取src作为prod canonical smiles参考
    canonical_list, class_mark_list = read_file(src_file, have_class=have_class, write_class=write_class)

    # 读取预测结果
    predict_list, _ = read_file(pre_file, have_class=have_class)
    predict_group_list = pre_list_to_group_list(predict_list, beam_size=beam_size)
    print('Read Done!')
    # 读取tgt作为ground truth
    if evaluation:
        ground_true_list, _ = read_file(tgt_file, have_class=have_class)
        assert len(canonical_list) == len(ground_true_list) == len(predict_group_list)
        if class_mark_list is not None:
            test_df = pd.DataFrame({'index': [i for i in range(len(ground_true_list))],
                                    'target': ground_true_list,
                                    'class_mark': class_mark_list})
        else:
            test_df = pd.DataFrame({'index': [i for i in range(len(ground_true_list))],
                                    'target': ground_true_list})

    else:
        print('Using...')
        assert len(canonical_list) == len(predict_group_list)
        test_df = pd.DataFrame({'index': [i for i in range(len(canonical_list))],
                                'target': ['' for i in range(len(canonical_list))]})
    pool = Pool(opt.core)
    total = len(predict_group_list)
    tasks = list(zip([i for i in range(total)], canonical_list, predict_group_list))
    all_results_list = []
    for result in tqdm(pool.imap_unordered(run_tasks, tasks), total=len(tasks)):
        all_results_list.append(result)

    all_results_list.sort(key=lambda x: x[0])
    for i, one_marked_list_rerank in tqdm(all_results_list):
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
            print('P2S Top-{}: {:.1f}%'.format(i, 100 * correct / total))
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
    parser.add_argument('-src_file', type=str, default='')
    parser.add_argument('-tgt_file', type=str, default='')
    parser.add_argument('-pre_file', type=str, default='')
    parser.add_argument('-save_rank_results_file', type=str, default='')
    parser.add_argument('-save_top', type=str, default='top_results.csv')
    parser.add_argument('-evaluation', action='store_true')
    parser.add_argument('-have_class', action='store_true')
    parser.add_argument('-write_to_step2', action='store_true')
    parser.add_argument('-write_class', action='store_true')
    parser.add_argument('-step2_save_file', type=str, default='')
    parser.add_argument('-step2_save_top1_file', type=str, default='')
    parser.add_argument('-core', type=int, default=8)
    opt = parser.parse_args()
    main(opt)
