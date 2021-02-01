from rdkit import Chem
import argparse

def canonical_smiles(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles),canonical=True)
def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    import re
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='smi_tokenizer.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-file', type=str, default='product_data/test.txt')
    opt = parser.parse_args()


    with open(opt.file,'r',encoding='utf-8') as f:
        file_list = [line.strip() for line in f.readlines()]
    canonical_list = [canonical_smiles(smiles) for smiles in file_list]
    with open(opt.file.replace('.txt','_canonical.txt'),'w',encoding='utf-8') as f:
        for smiles in canonical_list:
            f.write(smi_tokenizer(smiles)+'\n')

    
