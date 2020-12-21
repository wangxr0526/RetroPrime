import argparse
import os
import pandas as pd
import torch
from rdkit import Chem
from tqdm import tqdm
from retroprime.data_process.utiles import get_rxn_position_info_pd
from multiprocessing import Pool


def clear_map_to_canonical(smiles):
    mol = Chem.MolFromSmiles(smiles)
    [atom.ClearProp('molAtomMapNumber') for atom in mol.GetAtoms() if atom.HasProp('molAtomMapNumber')]
    return Chem.MolToSmiles(mol, True)


def run_task(task):
    index, rxn = task
    try:
        rxn_position_info_pd = get_rxn_position_info_pd(rxn)
        prod = rxn.split('>')[-1]
        canonical_smiles = clear_map_to_canonical(prod)
        return index, rxn_position_info_pd, canonical_smiles
    except:
        return index, None, None


if __name__ == '__main__':
    pool = Pool(8)
    # Designated type of reaction
    opt = argparse.ArgumentParser()
    opt.add_argument('-clean_data', default='../../databox/uspto_full/single/new_raw_all.csv')
    opt.add_argument('-output_dir', default='../../databox/uspto_full/single/index_save/')
    args, _ = opt.parse_known_args()
    database = pd.read_csv(args.clean_data)
    output_dir = args.output_dir
    rxn_smiles = database['reactants>reagents>production'].tolist()

    rxn_position_info_pd_list_index = []
    tasks = [(index, rxn) for index, rxn in enumerate(rxn_smiles)]
    for results in tqdm(pool.imap_unordered(run_task, tasks), total=len(tasks)):
        index, rxn_position_info_pd, canonical_smiles = results
        rxn_position_info_pd_list_index.append((index, rxn_position_info_pd, canonical_smiles))

    rxn_position_info_pd_list_index.sort(key=lambda x: x[0])

    rxn_position_info_pd_list = [x[1] for x in rxn_position_info_pd_list_index]

    torch.save(rxn_position_info_pd_list, os.path.join(output_dir, 'rxn_position_info_pd_list_good'))
    torch.save(rxn_position_info_pd_list_index, os.path.join(output_dir, 'rxn_position_info_pd_list_index_good'))

    drop_index = []
    rxn_position_info_pd_list_end = []
    prod_smiles = []
    for index, marked_prod, canonical_smiles in rxn_position_info_pd_list_index:
        if marked_prod is None:
            drop_index.append(index)
            prod_smiles.append(canonical_smiles)
        else:
            rxn_position_info_pd_list_end.append(marked_prod)
    # drop no atom changed reactions from raw datadf
    torch.save(rxn_position_info_pd_list_end, os.path.join(output_dir, '../rxn_position_info_pd_list_end'))
    print('drop {} reactions'.format(len(drop_index)))

    new_database = database.drop(drop_index)
    assert len(new_database['reactants>reagents>production']) == len(prod_smiles)
    new_database['prod_smiles'] = prod_smiles
    new_database.to_csv(os.path.join(output_dir, '../database_uspto_full_before_alignment.csv'), index=False)
