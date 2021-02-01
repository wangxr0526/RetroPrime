import os

import pandas as pd

if __name__ == '__main__':
    org_path = os.path.join('../../databox/select_50k', 'org_format_data')

    id = []
    dataset_mark = []
    rxn_smiles = []
    for dataset in ['train', 'val', 'test']:
        df = pd.read_csv(os.path.join(org_path, '{}.csv'.format(dataset)))
        id += df['id'].tolist()
        dataset_mark += ['{}'.format(dataset) for i in range(len(df.index))]
        rxn_smiles += df['rxn_smiles'].tolist()

    all_data_df = pd.DataFrame(
        {
            'id': id,
            'reactants>reagents>production': rxn_smiles,
            'dataset': dataset_mark
        }
    )
    all_data_df.to_csv(os.path.join(org_path, '../', 'new_raw_all.csv'), index=False)
