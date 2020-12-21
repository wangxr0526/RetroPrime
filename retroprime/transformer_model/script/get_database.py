import rdkit
import re
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import os
import sys
from tqdm import tqdm
from copy import deepcopy
#sys.path.append(r'E:/mpython/my_code/my_ai_Retrosynthetic/script')
#sys.path.append(r'E:/mpython/source_code_and_database/source_code/Computer Assisted Retrosynthesis Based on Molecular Similarity/retrosim-master/')
#import copy_generate_retro_templates as cgs
#from rdchiral.main import rdchiralReactants,rdchiralReaction,rdchiralRun
USE_STEREOCHEMISTRY = True

def seach(str):
    try:
        str.index('|')
        return str[int(str.index('|')):]
    except:
        return ''

def get_data_from_xml(path):
    '''
    :param path
    :return:reaction smiles list
    '''
    f = open(path, 'r', encoding='utf-8')
    a = f.read().split('<dl:reactionSmiles>')[1:]
    rxn = [i.split('</dl:reactionSmiles>')[0] for i in a]
    rxn_ = [rx.replace('&gt;','>') for rx in rxn]
    rxns = [rx.strip(seach(rx)) for rx in rxn_]
    f.close()
    return rxns

def get_temples_from_json(path):
    '''
    :param json path:
    :return: list(dict.items())
    '''
    import json
    with open(path,'r') as f:
        b = json.load(f)
    bitems = list(b.items())
    return bitems

def separate_temples_from_json(path):
    temples1_50 = []
    temples50_100 = []
    temples100_250 = []
    temples250_500 = []
    temples500_1000 = []
    temples1000_inf = []
    dict_items = get_temples_from_json(path)
    for i in range(len(dict_items)):
        if  dict_items[i][1]<=50:
            temples1_50.append(dict_items[i])
        if 50< dict_items[i][1]<=100:
            temples50_100.append(dict_items[i])
        if 100< dict_items[i][1]<=250:
            temples100_250.append(dict_items[i])
        if 250< dict_items[i][1]<=500:
            temples250_500.append(dict_items[i])
        if 500< dict_items[i][1]<=1000:
            temples500_1000.append(dict_items[i])
        if dict_items[i][1]>1000:
            temples1000_inf.append(dict_items[i])
    return temples1_50,temples50_100, temples100_250, temples250_500, temples500_1000, temples1000_inf


def get_map_mol(mol):
    '''
    去掉没有原子标号的原子，返回新的分子残片
    :param mol:
    :return:
    '''
    from rdkit.Chem import AllChem
    use_atom = []
    for atom in mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            use_atom.append(atom.GetIdx())
    return AllChem.MolFragmentToSmiles(mol, use_atom)

def get_nomap_mol(mol):
    '''
    去掉有原子标号的原子，返回新的分子残片
    :param mol:
    :return:
    '''
    from rdkit.Chem import AllChem
    use_atom = []
    for atom in mol.GetAtoms():
        if not atom.HasProp('molAtomMapNumber'):
            use_atom.append(atom.GetIdx())
    if len(use_atom) == 0:
        return ''
    else:
        return AllChem.MolFragmentToSmiles(mol, use_atom)

def remove_map(mol):
    for atom in mol.GetAtoms():
        atom.ClearProp('molAtomMapNumber')
    return mol

def temple_map_remove(temple,reverse=False):
    '''
    传入有原子map的smarts temple
    返回去除map的smiles temple
    :param temple:
    :return:
    '''
    from rdkit import Chem
    rt_smi = temple.split('>>')[0].split('.')
    pd_smi = temple.split('>>')[1].split('.')
    rt_mol = [Chem.MolFromSmarts(rt) for rt in rt_smi]
    pd_mol = [Chem.MolFromSmarts(pd) for pd in pd_smi]
    rt_mol_map_removed = [remove_map(rt) for rt in rt_mol]
    pd_mol_map_removed = [remove_map(pd) for pd in pd_mol]
    rt_smi_map_removed = [Chem.MolToSmiles(rt) for rt in rt_mol_map_removed]
    pd_smi_map_removed = [Chem.MolToSmiles(pd) for pd in pd_mol_map_removed]
    rt_smi_all = '.'.join(rt_smi_map_removed)
    pd_smi_all = '.'.join(pd_smi_map_removed)
    if not reverse:
        return '{}>>{}'.format(rt_smi_all, pd_smi_all)
    if reverse:
        return '{}>>{}'.format(pd_smi_all, rt_smi_all)


def get_strict_smarts_for_mol(mol,ref_smiles, allBondsExplicit=False):
    '''
    规范化反应物分子
    :param mol:
    :return:
    '''
    my_atom_smarts = MyGetAtom(ref_smiles)
    try:
        Chem.SanitizeMol(mol)
        Chem.AssignAtomChiralTagsFromStructure(mol)
        Chem.AssignStereochemistry(mol, flagPossibleStereoCenters=True)
        mol.UpdatePropertyCache()
        symbols = []
        atom_list = []
        for atom in mol.GetAtoms():
            symbols.append(cgs.get_strict_smarts_for_atom_test(atom,atom_smarts_from_smiles=my_atom_smarts[atom.GetIdx()]))
            atom_list.append(atom.GetIdx())
        return AllChem.MolFragmentToSmiles(mol,atom_list,atomSymbols=symbols,allHsExplicit=True, isomericSmiles=True, allBondsExplicit=allBondsExplicit)
    except Exception as e:
        print(e)
        return None

def get_strict_smarts_for_mol_test(mol,allBondsExplicit=False):
    '''
    规范化反应物分子
    :param mol:
    :return:
    '''

    try:
        Chem.SanitizeMol(mol)
        Chem.AssignAtomChiralTagsFromStructure(mol)
        Chem.AssignStereochemistry(mol, flagPossibleStereoCenters=True)
        mol.UpdatePropertyCache()
        symbols = []
        atom_list = []
        for atom in mol.GetAtoms():
            symbols.append(cgs.get_strict_smarts_for_atom_test(atom).replace(']','^{}]'.format(atom.GetIdx())))
            atom_list.append(atom.GetIdx())
        return AllChem.MolFragmentToSmiles(mol,atom_list,atomSymbols=symbols,allHsExplicit=True, isomericSmiles=True, allBondsExplicit=allBondsExplicit)
    except Exception as e:
        print(e)
        return None


def get_strict_smarts_for_mol_and_rm_unmap_atom(mol,san=False,allBondsExplicit=True):
    '''
    去除数据集反应物中的unmaped原子
    :param mol:
    :return:
    '''
    symbols = []
    atom_list = []
    try:
        Chem.SanitizeMol(mol)
        mol.UpdatePropertyCache()
        for atom in mol.GetAtoms():
            if atom.HasProp('molAtomMapNumber'):
                atom_list.append(atom.GetIdx())
                symbols.append(cgs.get_strict_smarts_for_atom(atom))
            elif not atom.HasProp('molAtomMapNumber'):
                symbols.append(atom.GetSmarts())
    except:
        if san:
            try:
                Chem.SanitizeMol(mol)
                mol.UpdatePropertyCache()
            except:return
        for atom in mol.GetAtoms():
            if atom.HasProp('molAtomMapNumber'):
                atom_list.append(atom.GetIdx())
                symbols.append(cgs.get_strict_smarts_for_atom(atom))
            elif not atom.HasProp('molAtomMapNumber'):
                symbols.append(atom.GetSmarts())

    return AllChem.MolFragmentToSmiles(mol, atom_list, atomSymbols=symbols,allHsExplicit=True, isomericSmiles=True, allBondsExplicit=allBondsExplicit)

def get_strict_smarts_for_mol_form_tp_and_rm_unmap_atom(mol,allBondsExplicit=True):
    '''
    去除来自于模板的反应物的unmaped原子
    :param mol:
    :return:
    '''
    symbols = []
    atom_list = []
    for atom in mol.GetAtoms():
        symbols.append(atom.GetSmarts())
        if atom.HasProp('molAtomMapNumber'):
            atom_list.append(atom.GetIdx())
    return AllChem.MolFragmentToSmiles(mol, atom_list, atomSymbols=symbols,allHsExplicit=True, isomericSmiles=True, allBondsExplicit=allBondsExplicit)


def is_number(s):
    try:
        float(s)
        return True
    except: return False

def have_Cl(smarts_list):
    for i in range(len(smarts_list)-1):
        if smarts_list[i] is 'C' and smarts_list[i+1] is 'l':
            smarts_list[i] = 'Cl'
            smarts_list[i+1] = ''

def have_Br(smarts_list):
    for i in range(len(smarts_list)-1):
        if smarts_list[i] is 'B' and smarts_list[i+1] is 'r':
            smarts_list[i] = 'Br'
            smarts_list[i+1] = ''

def split_smarts(smarts,check_rdkit=False,all_bond=False,debug=False):
    '''
    拆分smarts但将方括号里面的看作一个原子进行分词，且将环的标记看作一个词
    :param smarts:
    :return:
    '''
    try:
        if check_rdkit:
            try:
                mol = Chem.MolFromSmarts(smarts)
                Chem.SanitizeMol(mol)
            except:
                return print('not good smarts')
        atoms = re.findall('\[.*?\]', smarts)
        others = re.findall('\].*?\[', smarts)
        for i in range(len(others)):
            others[i] = others[i].replace(']','')
            others[i] = others[i].replace('[','')
        all_=[]
        for i in range(len(atoms)):
            all_.append(atoms[i])
            try:
                if others[i] is not '':
                    all_.append(others[i])
            except:continue
        if all_bond:
            all_.append(smarts[smarts.rindex(']')+1:])
        smarts_splited = []
        start_smarts = smarts[:smarts.index('[')]
        if start_smarts is not '':
            if is_number(start_smarts):
                smarts_splited.append(start_smarts)
            else:
                for e in start_smarts:
                    smarts_splited.append(e)
        for i in range(len(all_)):
            if '[' not in all_[i]:
                for j in range(len(all_[i])):
                    smarts_splited.append(all_[i][j])
            else:smarts_splited.append(all_[i])


        if not all_bond:
            end_smarts = smarts[smarts.rindex(']')+1:]
            if end_smarts is not '':
                if is_number(end_smarts):
                    smarts_splited.append(end_smarts)
                else:
                    for e in end_smarts:
                        smarts_splited.append(e)
        number_list = []
        for i in range(len(smarts_splited)):
            if is_number(smarts_splited[i]):
                number_list.append(i)
        for i in range(len(number_list)-1):

            if number_list[i]-number_list[i-1] == 1 :
                smarts_splited[number_list[i-1]] = smarts_splited[number_list[i-1]]+smarts_splited[number_list[i]]
                smarts_splited[number_list[i]] = ''
        if int(len(smarts_splited))in number_list and int(len(smarts_splited)-1) in number_list:
            smarts_splited[-2] = smarts_splited[-2]+smarts_splited[-1]
            smarts_splited[-1] = ''
        smarts_splited_ = []
        have_Br(smarts_splited)
        have_Cl(smarts_splited)
        for ss in smarts_splited:
            if ss is not '':
                smarts_splited_.append(ss)
        return smarts_splited_
    except Exception as e:
        if debug:
            print(e)
        return print('err')

def split_smiles(smiles):
    if '[' in smiles:
        return split_smarts(smiles)
    if '[' not in smiles:
        smiles_splited_list = [s for s in smiles]
        have_Cl(smiles_splited_list)
        have_Br(smiles_splited_list)
        smiles_splited = []
        for s in smiles_splited_list:
            if s != '':
                smiles_splited.append(s)
        return smiles_splited


def remove_err_atom_info(smarts):
    mol = Chem.MolFromSmarts(smarts)
    err = []
    try:
        Chem.SanitizeMol(mol)
        mol.UpdatePropertyCache()
    except Exception as e:
        err.append(e)
    if len(err) != 0:
        err_atom = re.findall(
            'Explicit valence for atom \# ([0-9]+) ([a-zA-Z]+)\, ([0-9]+)\, is greater than permitted', str(err[0]))
        symbol = []
        atom_list = []
        for atom in mol.GetAtoms():
            atom_list.append(atom.GetIdx())
            if atom.GetIdx() == int(err_atom[0][0]):
                symbol_change = re.sub('H[0-9]+[,&]D[0-9]+', '', atom.GetSmarts())
                symbol_change = re.sub('[+,-][0-9]','',symbol_change)
                symbol_change = re.sub('[;&]', '', symbol_change)
                symbol.append(symbol_change)
            else:
                symbol.append(atom.GetSmarts())
        return AllChem.MolFragmentToSmiles(mol, atom_list, atomSymbols=symbol, allHsExplicit=True, isomericSmiles=True, allBondsExplicit=True)
    else:
        return smarts


def remove_err_ring_atom_info(smarts_nobond):
    mol = Chem.MolFromSmarts(smarts_nobond)
    err = []
    symbol = []
    atom_list = []
    for atom in mol.GetAtoms():
        atom_list.append(atom.GetIdx())
        if len(re.findall('[a-z]',atom.GetSmarts()))!=0:
            symbol_change = re.sub('[;&]H0[;&]D[0-9][;&]\+0','',atom.GetSmarts())
            symbol_change = re.sub('[;&]H1[;&]D[0-9][;&]\+0','H',symbol_change)
            symbol_change = re.sub('[;&]H2[;&]D[0-9][;&]\+0','H2',symbol_change)
            symbol_change = re.sub('[;&]D[0-9][;&]\+0','',symbol_change)
            symbol.append(symbol_change)
        else:
            symbol.append(atom.GetSmarts())
    return AllChem.MolFragmentToSmiles(mol,atom_list,atomSymbols=symbol, allHsExplicit=True, isomericSmiles=True, allBondsExplicit=False)

def err_check(mol,err_print = False):
    try:
        Chem.SanitizeMol(mol)
        mol.UpdatePropertyCache()
        return True
    except Exception as e:
        if err_print:
            print(e)
        return False


def rm_NO2_info(smarts):
    '''
    只检测完整的NO2 smarts ，需要使用此函数最先除去所有NO2,此函数一定要在最先使用
    生成的smarts环上的原子smarts标记还得替换，不然会出错。
    '''
    mol = Chem.MolFromSmarts(smarts)
    symbol = []
    atom_list = []
    change_map_list = []
    NO2_atom_dir = {}
    for atom in mol.GetAtoms():
        '''检测NO2'''
        if len(re.findall('\[N[;&]\+[;&]H0[;&]D3', atom.GetSmarts())) == 1:
            NO2_atom_dir[atom.GetIdx()] = 'N+'
            for neighbor in atom.GetNeighbors():
                if len(re.findall('\[O[;&]H0[;&]D1[;&]\+0', neighbor.GetSmarts())) == 1:
                    NO2_atom_dir[neighbor.GetIdx()] = 'O'
                if len(re.findall('\[O[;&]\-[;&]H0[;&]D1', neighbor.GetSmarts())) == 1:
                    NO2_atom_dir[neighbor.GetIdx()] = 'O-'
    if len(NO2_atom_dir) != 0:
        for atom in mol.GetAtoms():
            '''替换smarts'''
            atom_list.append(atom.GetIdx())
            if atom.GetIdx() in NO2_atom_dir:
                if atom.HasProp('molAtomMapNumber'):
                    atom_map = atom.GetProp('molAtomMapNumber')
                    change_map_list.append(atom_map)
                    symbol.append('[{}:{}]'.format(NO2_atom_dir[atom.GetIdx()], atom_map))
                elif not atom.HasProp('molAtomMapNumber'):
                    symbol.append('[{}]'.format(NO2_atom_dir[atom.GetIdx()]))
            else:
                symbol.append(atom.GetSmarts())
        smarts = AllChem.MolFragmentToSmiles(mol, atom_list,
                                         atomSymbols=symbol,
                                         allHsExplicit=True,
                                         isomericSmiles=True,
                                         allBondsExplicit=False)
    else:
        smarts = smarts
    return smarts, change_map_list


def get_strict_smarts_for_atom_ignore_a_NO2(atom, super_general=False):
    '''
    For an RDkit atom object, generate a SMARTS pattern that
    matches the atom as strictly as possible
    '''
    symbol = atom.GetSmarts()
    if len(re.findall('[a-z]',symbol))==0:
        if atom.GetSymbol() == 'H':
            symbol = '[#1]'

        if '[' not in symbol:
            symbol = '[' + symbol + ']'

        # Explicit stereochemistry - *before* H
        if USE_STEREOCHEMISTRY:
            if atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
                if '@' not in symbol:
                    # Be explicit when there is a tetrahedral chiral tag
                    if atom.GetChiralTag() == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW:
                        tag = '@'
                    elif atom.GetChiralTag() == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW:
                        tag = '@@'
                    if ':' in symbol:
                        symbol = symbol.replace(':', ';{}:'.format(tag))
                    else:
                        symbol = symbol.replace(']', ';{}]'.format(tag))

        if 'H' not in symbol:
            H_symbol = 'H{}'.format(atom.GetTotalNumHs())
            # Explicit number of hydrogens
            if ':' in symbol: # stick H0 before label
                symbol = symbol.replace(':', ';{}:'.format(H_symbol))
            else:
                symbol = symbol.replace(']', ';{}]'.format(H_symbol))

        if not super_general:
            # Explicit degree
            if ':' in symbol:
                symbol = symbol.replace(':', ';D{}:'.format(atom.GetDegree()))
            else:
                symbol = symbol.replace(']', ';D{}]'.format(atom.GetDegree()))

        # Explicit formal charge
        if '+' not in symbol and '-' not in symbol:
            charge = atom.GetFormalCharge()
            charge_symbol = '+' if (charge >= 0) else '-'
            charge_symbol += '{}'.format(abs(charge))
            if ':' in symbol:
                symbol = symbol.replace(':', ';{}:'.format(charge_symbol))
            else:
                symbol = symbol.replace(']', ';{}]'.format(charge_symbol))
        if 'N' and '+' in atom.GetSmarts():#判断硝基的N+和季铵盐N+
            # ne = []
            # for neighbor in atom.GetNeighbors():
            #     if 'O' in neighbor.GetSmarts() and '-'not in neighbor.GetSmarts():
            #         ne.append('O')
            #     if 'O' and '-' in neighbor.GetSmarts():
            #         ne.append('O-')
            # if 'O' and 'O-' in ne:
            #     symbol = re.sub('\[N\&\+','[N+',atom.GetSmarts())
            symbol = re.sub('\[N\&\+', '[N+', atom.GetSmarts())
        if 'O' and '-' in atom.GetSmarts():#判断硝基的O-
            ne = []
            for neighbor in atom.GetNeighbors():
                if 'N' and '+' in neighbor.GetSmarts():
                    ne.append('N+')
            if 'N+' in ne:
                symbol = re.sub('\[O\&\-','[O-',atom.GetSmarts())
        if 'O' in atom.GetSmarts() and '-'not in atom.GetSmarts():#判断硝基的O
            ne = []
            for neighbor in atom.GetNeighbors():
                if 'N' and '+' in neighbor.GetSmarts():
                    ne.append('N+')
            if 'N+' in ne:
                symbol = atom.GetSmarts()
    elif len(re.findall('[a-z]',symbol))!=0:
        if 'H' in symbol:
            symbol = re.sub('[;&]H1','H',symbol)
            symbol = re.sub('[;&]H2','H2',symbol)
    return symbol

def get_strict_smarts_for_mol_ignore_a_NO2(mol,allBondsExplicit=False):
    '''
    规范化反应物分子
    :param mol:
    :return:
    '''


    Chem.SanitizeMol(mol)
    mol.UpdatePropertyCache()
    symbols = []
    atom_list = []
    for atom in mol.GetAtoms():
        symbols.append(get_strict_smarts_for_atom_ignore_a_NO2(atom))
        atom_list.append(atom.GetIdx())
    return AllChem.MolFragmentToSmiles(mol,atom_list,
                                       atomSymbols=symbols,
                                       allHsExplicit=True,
                                       isomericSmiles=True,
                                       allBondsExplicit=allBondsExplicit)

def self_initialize_reactants_from_smiles(smiles):
    reactants = Chem.MolFromSmiles(smiles)
    Chem.AssignStereochemistry(reactants, flagPossibleStereoCenters=True)
    reactants.UpdatePropertyCache()
    return reactants



def get_strict_smarts_for_mol_and_rm_unmap_atom_ignore_a_NO2(mol,allBondsExplicit=False):
    '''
    去除数据集反应物中的unmaped原子
    :param mol:
    :return:
    '''
    Chem.SanitizeMol(mol)
    mol.UpdatePropertyCache()
    symbols = []
    atom_list = []
    for atom in mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            atom_list.append(atom.GetIdx())
            symbols.append(get_strict_smarts_for_atom_ignore_a_NO2(atom))
        elif not atom.HasProp('molAtomMapNumber'):
            symbols.append(atom.GetSmarts())


    return AllChem.MolFragmentToSmiles(mol, atom_list,
                                       atomSymbols=symbols,
                                       allHsExplicit=True,
                                       isomericSmiles=True,
                                       allBondsExplicit=allBondsExplicit)


def bondtype2int(bondtype):
    lib = {'SINGLE':1,
           'DOUBLE':2}
    return lib[str(bondtype)]

def get_all_atom_neigbor_bond(mol,neighbor_idx,atom):
    atom_idx = atom.GetIdx()
    all_bond_num = 0
    for i in neighbor_idx:
        all_bond_num += bondtype2int(mol.GetBondBetweenAtoms(i,atom_idx).GetBondType())
    return all_bond_num

def get_edge_symbol(atom,all_bond_num):
    '''只用于无标记的smarts'''
    symbol = atom.GetSmarts()
    all_H_num = atom.GetTotalNumHs() + all_bond_num
    if '+' not in symbol and '-' not in symbol:
        if 'H' not in symbol:
            if all_H_num != 0:
                if all_H_num ==1:
                    H_symbol = 'H'
                else:
                    H_symbol = 'H{}'.format(all_H_num)
            elif all_H_num == 0:
                H_symbol = ''
            # Explicit number of hydrogens
            symbol = symbol.replace(':', '{}:'.format(H_symbol))
        if 'H' in symbol:
            if atom.GetTotalNumHs() == 1:
                if 'H:' in symbol:
                    symbol = re.sub('[&;]','',symbol)
                    symbol = re.sub('H','H{}'.format(all_H_num),symbol)
                else:
                    symbol = re.sub('[&;]','',symbol)
                    symbol = re.sub('H[0-9]','H{}'.format(all_H_num),symbol)
            if atom.GetTotalNumHs() != 1:
                symbol = re.sub('[&;]','',symbol)
                symbol = re.sub('H[0-9]','H{}'.format(all_H_num),symbol)
    if '+' in symbol:
        if 'H' not in symbol:
            if all_H_num != 0:
                if all_H_num ==1:
                    H_symbol = 'H'
                else:
                    H_symbol = 'H{}'.format(all_H_num)
            elif all_H_num == 0:
                H_symbol = ''
            # Explicit number of hydrogens
            symbol = symbol.replace('+', '{}+'.format(H_symbol))
        if 'H' in symbol:
            symbol = re.sub('[&;]','',symbol)
            symbol = re.sub('H[0-9]\+','H{}+'.format(all_H_num),symbol)
            symbol = re.sub('\+H[0-9]','H{}+'.format(all_H_num),symbol)
    if '-' in symbol:
        if 'H' not in symbol:
            if all_H_num != 0:
                if all_H_num ==1:
                    H_symbol = 'H'
                else:
                    H_symbol = 'H{}'.format(all_H_num)
            elif all_H_num == 0:
                H_symbol = ''
            # Explicit number of hydrogens
            symbol = symbol.replace('-', '{}-'.format(H_symbol))
        if 'H' in symbol:
            symbol = re.sub('[&;]','',symbol)
            symbol = re.sub('H[0-9]\-','H{}-'.format(all_H_num),symbol)
            symbol = re.sub('\-H[0-9]','H{}-'.format(all_H_num),symbol)
    return symbol

def get_strict_stable_smarts_for_mol_and_rm_unmap_atom_ignore_a_NO2(mol,allBondsExplicit=False):
    '''
    去除数据集反应物中的unmaped原子，只用于没有标记的原始smarts，标记过的会出错误
    :param mol:
    :return:
    '''
    Chem.SanitizeMol(mol)
    mol.UpdatePropertyCache()
    symbols = []
    atom_list = []
    for atom in mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            atom_list.append(atom.GetIdx())
            edge = False
            neighbor_idx = []
            for neighbor in atom.GetNeighbors():
                if not neighbor.HasProp('molAtomMapNumber'):
#                     neighbor_smarts = gd.get_strict_smarts_for_atom_ignore_a_NO2(neighbor)
                    neighbor_idx.append(neighbor.GetIdx())
                    edge = True
            if edge:
                all_bond_num = get_all_atom_neigbor_bond(mol,neighbor_idx,atom)
                symbol = get_edge_symbol(atom,all_bond_num)
                symbols.append(symbol)
            if not edge:
                symbols.append(get_strict_smarts_for_atom_ignore_a_NO2(atom))
        elif not atom.HasProp('molAtomMapNumber'):
            symbols.append(atom.GetSmarts())


    return AllChem.MolFragmentToSmiles(mol, atom_list,
                                       atomSymbols=symbols,
                                       allHsExplicit=True,
                                       isomericSmiles=True,
                                       allBondsExplicit=allBondsExplicit)

def smarts2smiles(test_smarts):

    test_split = test_smarts.split(' ')
    test_smarts_ = ''
    for a in test_split:
        test_smarts_ +=a
    test_smarts_sub = re.sub('\;D([0-9]+)\;\+0\]',']',test_smarts_)
    test_smarts_sub = re.sub('\;D([0-9]+)\]',']',test_smarts_sub)
    test_smarts_sub = test_smarts_sub.replace(';H3','H3')
    test_smarts_sub = test_smarts_sub.replace(';H2','H2')
    test_smarts_sub = test_smarts_sub.replace(';H1','H')
    test_smarts_sub = test_smarts_sub.replace(';H0','')
    return test_smarts_sub


def smarts_file2smiles_file(path):
    read_path = path
    save_path = path.replace('.txt','_smiles.txt')
    with open(read_path,'r',encoding='utf-8') as f:
        beam = f.readlines()
    beam_smiles_list = []
    for i, pr in tqdm(enumerate(beam)):
        pr_sub = smarts2smiles(pr)
        pr_mol = Chem.MolFromSmiles(pr_sub)
        try:
            pr_smiles = Chem.MolToSmiles(pr_mol,isomericSmiles=True)
            beam_smiles_list.append(pr_smiles)
        except:beam_smiles_list.append('err')
    with open(save_path,'w',encoding='utf-8') as sf:
        for j in beam_smiles_list:
            sf.write(j+'\n')




def MyGetAtom(smarts):
    '''
    获取atom smarts列表（因为rdkit自带的atom.GetSmarts()函数会使手性标记改变，编写这个函数是为了比对原子smarts，经过程序验证，本函数可以替代atom.GetSmarts(),不过返回的是atom smarts的列表。）
    :param smarts:
    :return:
    '''
    raw_list = split_smiles(smarts)
    for i in range(len(raw_list)):
        raw_list[i] = re.sub('[=,#,(,),.]','',raw_list[i])
        if is_number(raw_list[i]):
            raw_list[i] = re.sub('[0-9]+','',raw_list[i])
        if raw_list[i] is ':':
            raw_list[i] = raw_list[i].replace(':','')
        if raw_list[i] is '-':
            raw_list[i] = raw_list[i].replace('-','')
        if raw_list[i] is '/':
            raw_list[i] = raw_list[i].replace('/','')
        if raw_list[i] is '\\':
            raw_list[i] = raw_list[i].replace('\\','')
    atom_smarts_list = []
    for i in range(len(raw_list)):
        if raw_list[i] is not '':
            atom_smarts_list.append(raw_list[i])
    return atom_smarts_list


def smarts_file2smiles_file(path):
    read_path = path
    save_path = path.replace('.txt','_smiles.txt')
    with open(path,'r',encoding='utf-8') as f:
        beam5 = f.readlines()
    beam5_smiles_list = []
    for i, pr in tqdm(enumerate(beam5)):
        pr_sub = smarts2smiles(pr)
        pr_mol = Chem.MolFromSmiles(pr_sub)
        try:
            pr_smiles = Chem.MolToSmiles(pr_mol,isomericSmiles=True)
            beam5_smiles_list.append(pr_smiles)
        except:beam5_smiles_list.append('err')
#     beam5_smiles = pd.DataFrame(beam5_smiles_list)
#     return beam5_smiles.to_csv(save_path,index=False,header=None)
    with open(save_path,'w',encoding='utf-8') as sf:
        for j in beam5_smiles_list:
            sf.write(j+'\n')
def smarts2smiles(test_smarts):
#     test_smarts = '[O;H0;D1;+0] = [C;H0;D3;+0] 1 [C;H2;D2;+0] [C;H2;D2;+0] [C;H0;D3;+0] ( = [O;H0;D1;+0] ) [N;H0;D3;+0] 1 [Br] . [CH3;D1;+0] [CH2;D2;+0] [O;H0;D2;+0] [C;H0;D3;+0] ( = [O;H0;D1;+0] ) [c] 1 [n] [n] ( - [c] 2 [cH] [cH] [c] ( [Cl] ) [cH] [c] 2 [Cl] ) [c] ( - [c] 2 [cH] [cH] [c] ( [O;H0;D2;+0] [CH3;D1;+0] ) [cH] [cH] 2 ) [c] 1 [CH3;D1;+0]'
    test_split = test_smarts.split(' ')
    test_smarts_ = ''
    for a in test_split:
        test_smarts_ +=a
    test_smarts_sub = re.sub('\;D([0-9]+)\;\+0\]',']',test_smarts_)
    test_smarts_sub = re.sub('\;D([0-9]+)\]',']',test_smarts_sub)
# print(test_smarts_sub)
    test_smarts_sub = test_smarts_sub.replace(';H3','H3')
    test_smarts_sub = test_smarts_sub.replace(';H2','H2')
    test_smarts_sub = test_smarts_sub.replace(';H1','H')
    test_smarts_sub = test_smarts_sub.replace(';H0','')
    return test_smarts_sub


def get_rxn_centre2idx(map_smiles):
    centre_index = []
    centre_map = []
    mol = Chem.MolFromSmiles(map_smiles)
    for atom in mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            for nei in atom.GetNeighbors():
                if not nei.HasProp('molAtomMapNumber'):
                    centre_index.append(atom.GetIdx())
                    centre_index.append(nei.GetIdx())
                    centre_map.append(atom.GetProp('molAtomMapNumber'))
    if len(centre_index) == 0:
        return None
    return centre_index, centre_map

def get_rt_centre_map_from_pd(map_pd_smiles,use_map_list):
    mol = Chem.MolFromSmiles(map_pd_smiles)
    other_use_map = []
    for atom in mol.GetAtoms():
        if atom.GetProp('molAtomMapNumber') in use_map_list:
            for nei in atom.GetNeighbors():
                other_use_map.append(nei.GetProp('molAtomMapNumber'))
    return other_use_map

def get_all_map_rt_from_pd_map(all_map_rt_smiles,other_use_map):
    centre_index = []
    mol = Chem.MolFromSmiles(all_map_rt_smiles)
    for atom in mol.GetAtoms():
        if atom.GetProp('molAtomMapNumber') in other_use_map:
            centre_index.append(atom.GetIdx())
    return centre_index

def get_centre_indx_from_rxn(maped_rxn_smiles):
    '''
    输入带map的smiles化学方程式，输出反应中心的原子index，除了两个反应物原子都会出现在生成物中都可以处理。
    :param maped_rxn_smiles:
    :return:
    '''
    rts = maped_rxn_smiles.split('>>')[0].split('.')
    pds = maped_rxn_smiles.split('>>')[-1].split('.')
    if len(pds) != 1:
        print('Warning:pds num !=1')
    rts_index_flag_F = [get_rxn_centre2idx(rt) for rt in rts]
#     print(rts_index_flag_F)
    if None not in rts_index_flag_F:
        return [t[0] for t in rts_index_flag_F]
    if None in rts_index_flag_F:
        if len(rts_index_flag_F) >= 3:
            print('rt >2')
            return [[]]
        if len(rts_index_flag_F) <= 2:
            use_map_list = []
            use_map_indx = None
            for i,l in enumerate(rts_index_flag_F):
                if l is not None:
                    use_map_list = l[1]

                if l is None:
                    use_map_indx = i
            other_use_map = get_rt_centre_map_from_pd(pds[0],use_map_list)
            other_use_idx = get_all_map_rt_from_pd_map(rts[use_map_indx],other_use_map)
            rts_index_flag_E = deepcopy(rts_index_flag_F)
            for i in range(len(rts_index_flag_E)):
                if rts_index_flag_E[i] is None:
                    rts_index_flag_E[i] = (other_use_idx,None)
            return [t[0] for t in rts_index_flag_E]


def check_index(centre_index):
    '''
    输入函数get_centre_indx_from_rxn()的输出，如果其为空的话返回False
    :param centre_index:
    :return:
    '''
    all_len_idx = 0
    for ls in centre_index:
        all_len_idx += len(ls)
    if all_len_idx == 0:
        return False
    else:
        return True

def get_map_from_pattern_pd(mapped_rxn_smiles):
    template = cgs.process_an_example(mapped_rxn_smiles,super_general=True)
    template_pd = template.split('>>')[0]
    pattern_mol = Chem.MolFromSmarts(template_pd)
    pds_smiles = mapped_rxn_smiles.split('>>')[-1]
    mol_pds = Chem.MolFromSmiles(pds_smiles)
    centre_index = mol_pds.GetSubstructMatches(pattern_mol)
    use_map = []
    for t in centre_index:
        for idx in t:
            atom = mol_pds.GetAtomWithIdx(idx)
            map_num = atom.GetProp('molAtomMapNumber')
            use_map.append(map_num)
    return use_map

def get_centre_index_pattern_method(mapped_rxn_smiles):
    use_map = get_map_from_pattern_pd(mapped_rxn_smiles)
    rts = mapped_rxn_smiles.split('>>')[0].split('.')
    use_index_list = []
    for rt in rts:
        use_index = []
        rt_mol = Chem.MolFromSmiles(rt)
        for atom in rt_mol.GetAtoms():
            if atom.HasProp('molAtomMapNumber'):
                if atom.GetProp('molAtomMapNumber') in use_map:
                    use_index.append(atom.GetIdx())
                    for nei in atom.GetNeighbors():
                        if not nei.HasProp('molAtomMapNumber'):
                            use_index.append(nei.GetIdx())
        use_index_list.append(use_index)
    return use_index_list


def c2apbp(info_pds_smiles):
    '''
    拆解反应位点标1和4的分子，返回分子碎片（canonical）
    '''
    mol = Chem.MolFromSmiles(info_pds_smiles)
    maped_atom_index = []
    for atom in mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            if atom.GetProp('molAtomMapNumber') in ['1','4']:
                maped_atom_index.append(atom.GetIdx())
    bonds_id = []
    maped_atom_index_copy = deepcopy(maped_atom_index)
    if len(maped_atom_index) == 1:
        return info_pds_smiles
    for x in maped_atom_index:
        for y in maped_atom_index_copy:
            if mol.GetBondBetweenAtoms(x,y) is not None:
                bonds_id.append(mol.GetBondBetweenAtoms(x,y).GetIdx())
    set_bonds_id = list(set(bonds_id))
    if len(set_bonds_id) == 0:
        return info_pds_smiles
    frags = Chem.FragmentOnBonds(mol,set_bonds_id)
    frags_smiles = Chem.MolToSmiles(frags,canonical=True)
    frags_smiles_sub = re.sub('\[([0-9]+)\*\]','*',frags_smiles)
    if Chem.MolFromSmiles(frags_smiles_sub) is None:
        return Chem.MolToSmiles(Chem.MolFromSmarts(frags_smiles_sub),canonical=True)
    frags_end = Chem.MolToSmiles(Chem.MolFromSmiles(frags_smiles_sub),canonical=True)
    return frags_end

class get_split_bond_atom:
    def __init__(self,rxn_smiles):
        self.rxn_smiles = rxn_smiles
    def get_changed_map(self):
        rxn_smiles = self.rxn_smiles
        changed_map_all = cgs.get_changed_atom_map_from_rxn_smiles(rxn_smiles)
        changed_map_dic = {}
        changed_map_dic['pds'] = changed_map_all
        rts = rxn_smiles.split('>>')[0].split('.')
        for i,rt in enumerate(rts):
            changed_map_dic['rt{}'.format(i)] = []
            for map_ in changed_map_dic['pds']:
                if ':{}]'.format(map_) in rt:
                    changed_map_dic['rt{}'.format(i)].append(map_)
        return changed_map_dic
    def get_neighbor_map(self):
        rxn_smiles = self.rxn_smiles
        changed_map_dic = self.get_changed_map()
        mol_pds = Chem.MolFromSmiles(rxn_smiles.split('>>')[-1])
        pds_nei_dic = {}
        for atom in mol_pds.GetAtoms():
            if atom.GetProp('molAtomMapNumber') in changed_map_dic['pds']:
                pds_nei_dic[atom.GetProp('molAtomMapNumber')] = []
                for nei in atom.GetNeighbors():
                    pds_nei_dic[atom.GetProp('molAtomMapNumber')].append(nei.GetProp('molAtomMapNumber'))
        rts = rxn_smiles.split('>>')[0].split('.')
        rts_nei_dic = {}
        for i,rt in enumerate(rts):
            mol_rt = Chem.MolFromSmiles(rt)
            for atom in mol_rt.GetAtoms():
                if atom.HasProp('molAtomMapNumber'):
                    if atom.GetProp('molAtomMapNumber') in changed_map_dic['pds']:
                        rts_nei_dic[atom.GetProp('molAtomMapNumber')] = []
                        for nei in atom.GetNeighbors():
                            if nei.HasProp('molAtomMapNumber'):
                                rts_nei_dic[atom.GetProp('molAtomMapNumber')].append(nei.GetProp('molAtomMapNumber'))
                            if not nei.HasProp('molAtomMapNumber'):
                                rts_nei_dic[atom.GetProp('molAtomMapNumber')].append('0')
        return pds_nei_dic,rts_nei_dic
    def get_nei_diff(self):
        pds_nei_dic,rts_nei_dic = self.get_neighbor_map()
        split_bond_atom_map = []
        for map_ in pds_nei_dic.keys():
            if set(pds_nei_dic[map_]) != set(rts_nei_dic[map_]):
#                 if len(list(set(pds_nei_dic[map_]))) < len(list(set(rts_nei_dic[map_]))):
                split_bond_atom_map.append(map_)
        return split_bond_atom_map
    def is_self(self):
        pds_nei_dic,rts_nei_dic = self.get_neighbor_map()
        set_pds_nei_dic = {}
        for key in pds_nei_dic.keys():
            set_pds_nei_dic[key] = set(pds_nei_dic[key])
        set_rts_nei_dic = {}
        for key in rts_nei_dic.keys():
            set_rts_nei_dic[key] = set(rts_nei_dic[key])
        if set_pds_nei_dic == set_rts_nei_dic:
            return True
        else: return False


def get_mark_apbp(canonical_pos_info_pd):
    #传入的标记都需要只有一种
    mark = re.findall('\:([0-9]+)\]',canonical_pos_info_pd)[0]
    if mark == '1':
        split = True
        test_mol = Chem.MolFromSmiles(c2apbp(canonical_pos_info_pd))
        rxn_pos_index = []
        for atom in test_mol.GetAtoms():
            if atom.HasProp('molAtomMapNumber'):
                rxn_pos_index.append(atom.GetIdx())
        for atom in test_mol.GetAtoms():
            if atom.GetIdx() in rxn_pos_index:
                atom.SetProp('molAtomMapNumber','1')
                for nei in atom.GetNeighbors():
                    if nei.GetIdx() not in rxn_pos_index:
                        nei.SetProp('molAtomMapNumber','2')
        for atom in test_mol.GetAtoms():
            if not atom.HasProp('molAtomMapNumber'):
                atom.SetProp('molAtomMapNumber','3')
        smiles_apbp_info = Chem.MolToSmiles(test_mol,canonical=False)
        return re.sub('\[\*\:([0-9]+)\]','',smiles_apbp_info),split
    if mark == '4':
        split = True
        test_mol = Chem.MolFromSmarts(c2apbp(canonical_pos_info_pd))
        rxn_pos_index = []
        for atom in test_mol.GetAtoms():
            if atom.HasProp('molAtomMapNumber'):
                rxn_pos_index.append(atom.GetIdx())
        for atom in test_mol.GetAtoms():
            if atom.GetIdx() in rxn_pos_index:
                atom.SetProp('molAtomMapNumber','1')
                for nei in atom.GetNeighbors():
                    if nei.GetIdx() not in rxn_pos_index:
                        nei.SetProp('molAtomMapNumber','2')
        for atom in test_mol.GetAtoms():
            if not atom.HasProp('molAtomMapNumber'):
                atom.SetProp('molAtomMapNumber','3')
        smiles_apbp_info = Chem.MolToSmiles(test_mol,canonical=True)
        smiles_apbp_info_sub = re.sub('\(\[\*\:([0-9]+)\]\)','',smiles_apbp_info)
        smiles_apbp_info_sub1 = re.sub('\[\*\:([0-9]+)\]','',smiles_apbp_info_sub)
        return re.sub('\([=,-]\)','',smiles_apbp_info_sub1),split
    if mark in ['2','3']:
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
        if len(nei1_pos_index_list) != 0:
            frist_atom_index = nei1_pos_index_list[0]
        smiles1 = Chem.MolToSmiles(test_mol,rootedAtAtom=frist_atom_index,canonical=True)
        test_mol1 = Chem.MolFromSmiles(smiles1)
        rxn_pos_index = []
        for atom in test_mol1.GetAtoms():
            if atom.HasProp('molAtomMapNumber'):
                rxn_pos_index.append(atom.GetIdx())
        for atom in test_mol1.GetAtoms():
            if atom.GetIdx() in rxn_pos_index:
                atom.SetProp('molAtomMapNumber','1')
                for nei in atom.GetNeighbors():
                    if nei.GetIdx() not in rxn_pos_index:
                        nei.SetProp('molAtomMapNumber','2')
        for atom in test_mol1.GetAtoms():
            if not atom.HasProp('molAtomMapNumber'):
                atom.SetProp('molAtomMapNumber','3')
        smiles_apbp_info1 = Chem.MolToSmiles(test_mol1,canonical=True)
        return smiles_apbp_info1,split


def editdistance(str1, str2):
    # 计算编辑距离
    edit = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]

    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                d = 0
            else:
                d = 1
            edit[i][j] = min(edit[i - 1][j] + 1, edit[i][j - 1] + 1, edit[i - 1][j - 1] + d)
    return edit[len(str1)][len(str2)]

def get_mark_ab(rxn):
    rts = rxn.split('>>')[0]
    split_bond_class = get_split_bond_atom(rxn)
    mol_rts = Chem.MolFromSmiles(rts)
    if not split_bond_class.is_self():
        rxn_map_pos_list = split_bond_class.get_nei_diff()
    if split_bond_class.is_self():
        rxn_map_pos_list = cgs.get_changed_atom_map_from_rxn_smiles(rxn)
    rxn_pos_index = []
    for atom in mol_rts.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            if atom.GetProp('molAtomMapNumber') in rxn_map_pos_list:
                rxn_pos_index.append(atom.GetIdx())
    nei_index = []
    for atom in mol_rts.GetAtoms():
        if atom.GetIdx() in rxn_pos_index:
            atom.SetProp('molAtomMapNumber','1')
            for nei in atom.GetNeighbors():
                if nei.HasProp('molAtomMapNumber'):
                    if nei.GetIdx() not in rxn_pos_index:
                        nei_index.append(nei.GetIdx())
                        nei.SetProp('molAtomMapNumber','2')
    D_index = []
    for atom in mol_rts.GetAtoms():
        if not atom.HasProp('molAtomMapNumber'):
            D_index.append(atom.GetIdx())
            atom.SetProp('molAtomMapNumber','1')
    for atom in mol_rts.GetAtoms():
        if atom.GetIdx() not in rxn_pos_index+nei_index+D_index:
            atom.SetProp('molAtomMapNumber','3')
    ab_rt_smiles = Chem.MolToSmiles(mol_rts,canonical=True)
    return ab_rt_smiles

def get_iso_smiles(smiles):
    iso = Chem.MolToSmiles(Chem.MolFromSmiles(smiles),isomericSmiles=True)
    return iso
def info_smarts2smiles(smiles):
    smiles = ''.join(smiles.split(' '))
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        for atom in mol.GetAtoms():
            if atom.HasProp('molAtomMapNumber'):
                atom.ClearProp('molAtomMapNumber')
        canonical_smiles = Chem.MolToSmiles(mol,canonical=True)
        return canonical_smiles
    else:return 'err'
def info_smarts_file2smiles_file(path):
    save_path = path.replace('.txt','_smiles.txt')
    with open(path,'r',encoding='utf-8') as f:
        lines = f.readlines()
    smiles_list = []
    for line in tqdm(lines):
        iso_smiles = info_smarts2smiles(line)
        smiles_list.append(iso_smiles)
    with open(save_path,'w',encoding='utf-8') as f:
        for smiles in smiles_list:
            f.write(smiles+'\n')


def get_mark_c(canonical_pos_info_pd):
    mol = Chem.MolFromSmiles(canonical_pos_info_pd)
    rxn_pos_index = []
    for atom in mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            rxn_pos_index.append(atom.GetIdx())
    for atom in mol.GetAtoms():
        if atom.GetIdx() in rxn_pos_index:
            atom.SetProp('molAtomMapNumber','1')
            for nei in atom.GetNeighbors():
                if nei.GetIdx() not in rxn_pos_index:
                    nei.SetProp('molAtomMapNumber','2')
    for atom in mol.GetAtoms():
        if not atom.HasProp('molAtomMapNumber'):
            atom.SetProp('molAtomMapNumber','3')
    smiles = Chem.MolToSmiles(mol,canonical=True)
    return smiles

def atom_is_rough_same(atom1,atom2):
    atom_mol1 = Chem.MolFromSmarts(atom1)
    atom_mol_list1 = [a for a in atom_mol1.GetAtoms()]
    atom1 = atom_mol_list1[0]
    atom_mol2 = Chem.MolFromSmarts(atom2)
    atom_mol_list2 = [a for a in atom_mol2.GetAtoms()]
    atom2 = atom_mol_list2[0]
    if atom1.GetAtomicNum() != atom2.GetAtomicNum():return False
    if atom1.GetIsAromatic() != atom2.GetIsAromatic():return False
    return True

def get_info_num(atom_smarts):
    num = re.findall('\:([0-9]+)',atom_smarts)
    return num

def Execute_grammar_err(canonical_smiles,pre_smiles):
    canonical_smiles = canonical_smiles
    top1_pre = pre_smiles
    pre_atom_list = MyGetAtom(top1_pre)
    canonical_atom_list = MyGetAtom(canonical_smiles)
    flag = False
    Execute = False
    if len(MyGetAtom(canonical_smiles)) == len(MyGetAtom(top1_pre)):
        for k, atom_smiles in enumerate(canonical_atom_list):
            # info_num = get_info_num(pre_atom_list[k])
            try:
                if not atom_is_rough_same(atom_smiles, pre_atom_list[k]):
                    flag = True
            except:flag = True
        if not flag:
            mark_index = []
            mark_ = []
            for k, atom_smiles in enumerate(canonical_atom_list):
                info_num = get_info_num(pre_atom_list[k])
                if len(info_num) != 0:
                    mark_index.append(k)
                    mark_.append(info_num[0])
            if len(set(mark_)) != 1:
                Execute = True
                mark = None
            if len(set(mark_)) == 1:
                mark = mark_[0]
            if mark == '1':
                if len(mark_index) != 2:
                    Execute = True
            mol = Chem.MolFromSmiles(canonical_smiles)
            bond_list = []
            for x in mark_index:
                for y in mark_index:
                    bond = mol.GetBondBetweenAtoms(x, y)
                    if bond is not None:
                        bond_list.append(bond)
            if len(bond_list) == 0:
                if mark == '1':
                    Execute = True

        if flag:
            Execute = True
    if len(MyGetAtom(canonical_smiles)) != len(MyGetAtom(top1_pre)):
        Execute = True
    return Execute


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


def evaluate_c2c(root_dir, beam_size=10, step=90000):
    predict_dir = root_dir + 'predictions_USPTO-50K_model_step_{}.pt_on_USPTO-50K_beam10.txt'.format(step)
    ground_true_dir = root_dir + 'tgt-test.txt'
    canonical_dir = root_dir + 'src-test.txt'

    # 读取文件
    with open(predict_dir, 'r', encoding='utf-8') as f:
        predict_list_class = [line.replace(' ', '').replace('\n', '') for line in f.readlines()]
        predict_list = [re.sub('\<RC_([0-9]+)\>', '', line) for line in predict_list_class]
    with open(ground_true_dir, 'r', encoding='utf-8') as f:
        ground_true_list_class = [line.replace(' ', '').replace('\n', '') for line in f.readlines()]
        ground_true_list = [re.sub('\<RC_([0-9]+)\>', '', line) for line in ground_true_list_class]
    with open(canonical_dir, 'r', encoding='utf-8') as f:
        canonical_list = [line.replace(' ', '').replace('\n', '') for line in f.readlines()]
        canonical_no_class_list = [re.sub('\<RC_([0-9]+)\>', '', line) for line in canonical_list]

    # 预测数据按beam size分组
    print('all predict data:', len(predict_list))
    predict_group_list = []
    i = 0
    group = []
    for pre in predict_list:
        i += 1
        group.append(pre)
        if i == beam_size:
            i = 0
            predict_group_list.append(group)
            group = []

    # 先排除语法错误的
    # 收集语法正确的但是不和ground truth对应的数据，以字典形式储存{class+canonical smiles:label smiles}
    counta_true = []
    counta_false = []
    alla_count = 0
    err_to_data_aug_dic = {}
    for i, group in enumerate(predict_group_list):
        alla_count += 1
        new_group = []
        for pre in group[:]:
            if not Execute_grammar_err(canonical_no_class_list[i], pre):
                new_group.append(pre)
                #             if len(new_group) == 2:
                #                 break
                break
        for new_pre in new_group[:1]:
            if Chem.MolFromSmiles(new_pre) is not None:
                if get_info_index(new_pre) == get_info_index(ground_true_list[i]):
                    if i not in counta_true:
                        counta_true.append(i)
                if get_info_index(new_pre) != get_info_index(ground_true_list[i]):
                    mark_group = get_info_index(new_pre)
                    marked_aug_smiles = mark_canonical_from_mark(canonical_no_class_list[i], mark_group)
                    err_to_data_aug_dic[canonical_list[i]] = marked_aug_smiles
    print('top-{} {:.2f}%'.format(1, 100 * len(set(counta_true)) / alla_count))
    return counta_true

