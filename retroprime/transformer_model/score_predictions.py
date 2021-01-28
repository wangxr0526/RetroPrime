#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals
import argparse
from rdkit import Chem
import pandas as pd
import numpy as np
import onmt.opts
from tqdm import tqdm

import sys
def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    else:
        return ''

def canonicalize_smiles_clear_map(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        [atom.ClearProp('molAtomMapNumber') for atom in mol.GetAtoms() if atom.HasProp('molAtomMapNumber')]
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    else:
        return ''

def get_rank(row, base, max_rank):
    for i in range(1, max_rank+1):
        if row['target'] == row['{}{}'.format(base, i)]:
            return i
    return 0

def get_rerank(row, base, reranking, max_rank):
    for i in range(1, max_rank+1):
        if row['target'] == row['{}{}'.format(base, i)]:
            return row['{}{}'.format(reranking,i)]
    return 0


def main(opt):
    print('Reading targets from file ...')
    with open(opt.targets, 'r') as f:
        targets = [canonicalize_smiles_clear_map(''.join(line.strip().split(' '))) for line in tqdm(f.readlines())]

    targets = targets[:]
    predictions = [[] for i in range(opt.beam_size)]
    # print(targets[:3])
    # ['C1=COCCC1.COC(=O)CCC(=O)c1ccc(O)cc1O', 'COC(=O)c1cccc(C(=O)O)c1.Nc1cccnc1N', 'CC(C)(C)OC(=O)NC1CCC(C(=O)O)CC1.CNOC']

    test_df = pd.DataFrame(targets)
    test_df.columns = ['target']
    # search_results = []
    # for t in tqdm(test_df['target']): search_results.append([do_one(t)])
    # search_results = np.array([do_one(t) for t in test_df['target']])
    # print([len(x) for x in search_results])

    total = len(test_df)

    print('Reading predictions from file ...')
    with open(opt.predictions, 'r') as f:
        # lines = f.readlines()
        # lines = [''.join(x.strip().split()[1:]) for x in lines]
        # print(lines[1])
        for i, line in enumerate(f.readlines()):
            # if i ==800*10:
            #     break
            predictions[i % opt.beam_size].append(''.join(line.strip().split(' ')))
    
    print('Computing ranks ...')
    reranking_score = np.zeros(shape=[total, opt.beam_size])
    # for i, preds in enumerate(predictions):
    for i, preds in enumerate(tqdm(predictions)):
        canonical_preds = [canonicalize_smiles_clear_map(x) for x in preds]
        validity_score = np.array([not (x == '') for x in canonical_preds])
        if opt.rerank:
            reranking_score[:,i] = validity_score/(i+1)
        else:
            reranking_score[:,i] = 1./(i+1)
        test_df['prediction_{}'.format(i + 1)] = preds
        test_df['canonical_prediction_{}'.format(i + 1)] = canonical_preds
    
    # reranking predictions
    print('Computing reranks ...')
    reranking_args = np.zeros(shape=[total, opt.beam_size], dtype = int)
    for s in range(total):
        reranking_args[s,:] = (-reranking_score[s,:]).argsort()+1 

    def get_rerank_can_pred(row, pred, rerank, i):
        idx = row['{}{}'.format(rerank,i+1)]
        return row['{}{}'.format(pred, idx)]
    for i in range(opt.beam_size): 
        test_df['reranking_args_{}'.format(i+1)] = reranking_args[:,i]
        test_df['reranked_can_pred_{}'.format(i+1)] = test_df.apply(lambda row: get_rerank_can_pred(row, 'canonical_prediction_', 'reranking_args_', i), axis = 1)


    # if opt.invalid_smiles:
    #     test_df['rank'] = test_df.apply(lambda row: get_rank(row, 'canonical_prediction_', opt.beam_size), axis=1)
    # else:
    #     test_df['rank'] = test_df.apply(lambda row: get_rank(row, 'prediction_', opt.beam_size), axis=1)
    test_df['rank'] = test_df.apply(lambda row: get_rank(row, 'reranked_can_pred_', opt.beam_size), axis=1)

    test_df.to_csv('surprise.csv')
    correct = 0
    invalid_smiles = 0
    print('Results:\n')
    # for i in range(1, (opt.beam_size+1)//2 + 1):
    for i in range(1, opt.beam_size + 1):
        correct += (test_df['rank'] == i).sum()
        # invalid_smiles += (test_df['canonical_prediction_{}'.format(i)] == '').sum()
        invalid_smiles += (test_df['reranked_can_pred_{}'.format(i)] == '').sum()
        if opt.invalid_smiles:
            print('Top-{}: {:.1f}% || Invalid SMILES {:.2f}%'.format(i, correct/total*100,
                                                                     invalid_smiles/(total*i)*100))
        else:
            print('Top-{}: {:.1f}%'.format(i, correct / total * 100))




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='score_predictions.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    onmt.opts.add_md_help_argument(parser)

    parser.add_argument('-beam_size', type=int, default=10,
                       help='Beam size')
    parser.add_argument('-invalid_smiles', action="store_true",
                       help='Show %% of invalid SMILES')
    parser.add_argument('-predictions', type=str, default="experiments/final/predictions_on_Dataset_ClusterSplit_withRX_beam10.txt",
                       help="Path to file containing the predictions")
    parser.add_argument('-targets', type=str, default="data/MIT_mixed_augm_clusterSplit/tgt-valid-RX",
                       help="Path to file containing targets")

    opt = parser.parse_args()


#    opt.beam_size = 10
    opt.invalid_smiles = True
#    opt.predictions = 'experiments/results/predictions_USPTO-50K_model_step_160000.pt_on_USPTO-50K_beam10_smiles.txt'
#    opt.targets = 'data/USPTO-50K/tgt-test_smiles.txt'

    opt.rerank = True

    main(opt)


