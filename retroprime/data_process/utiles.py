'''
This script is changed from https://github.com/connorcoley/retrosim/blob/master/retrosim/utils/generate_retro_templates.py and added new features.
'''
import re
from copy import deepcopy
from itertools import permutations

from rdkit import Chem
from rdkit.Chem import AllChem

v = False
USE_STEREOCHEMISTRY = True


def mols_from_smiles_list(all_smiles):
    '''Given a list of smiles strings, this function creates rdkit
    molecules'''
    mols = []
    for smiles in all_smiles:
        if not smiles: continue
        mols.append(Chem.MolFromSmiles(smiles))
    return mols


def get_tagged_atoms_from_mol(mol):
    '''Takes an RDKit molecule and returns list of tagged atoms and their
    corresponding numbers'''
    atoms = []
    atom_tags = []
    for atom in mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            atoms.append(atom)
            atom_tags.append(str(atom.GetProp('molAtomMapNumber')))
    return atoms, atom_tags


def get_tagged_atoms_from_mols(mols):
    '''Takes a list of RDKit molecules and returns total list of
    atoms and their tags'''
    atoms = []
    atom_tags = []
    for mol in mols:
        new_atoms, new_atom_tags = get_tagged_atoms_from_mol(mol)
        atoms += new_atoms
        atom_tags += new_atom_tags
    return atoms, atom_tags


def bond_to_label(bond):
    '''This function takes an RDKit bond and creates a label describing
    the most important attributes'''
    # atoms = sorted([atom_to_label(bond.GetBeginAtom().GetIdx()), \
    #               atom_to_label(bond.GetEndAtom().GetIdx())])
    a1_label = str(bond.GetBeginAtom().GetAtomicNum())
    a2_label = str(bond.GetEndAtom().GetAtomicNum())
    if bond.GetBeginAtom().HasProp('molAtomMapNumber'):
        a1_label += bond.GetBeginAtom().GetProp('molAtomMapNumber')
    if bond.GetEndAtom().HasProp('molAtomMapNumber'):
        a2_label += bond.GetEndAtom().GetProp('molAtomMapNumber')
    atoms = sorted([a1_label, a2_label])

    return '{}{}{}'.format(atoms[0], bond.GetSmarts(), atoms[1])


def atoms_are_different(atom1, atom2):
    '''Compares two RDKit atoms based on basic properties'''

    if atom1.GetSmarts() != atom2.GetSmarts(): return True  # should be very general
    if atom1.GetAtomicNum() != atom2.GetAtomicNum(): return True  # must be true for atom mapping
    if atom1.GetTotalNumHs() != atom2.GetTotalNumHs(): return True
    if atom1.GetFormalCharge() != atom2.GetFormalCharge(): return True
    if atom1.GetDegree() != atom2.GetDegree(): return True
    # if atom1.IsInRing() != atom2.IsInRing(): return True
    if atom1.GetNumRadicalElectrons() != atom2.GetNumRadicalElectrons(): return True
    if atom1.GetIsAromatic() != atom2.GetIsAromatic(): return True
    # TODO: add # pi electrons like ICSynth?

    # Check bonds and nearest neighbor identity
    bonds1 = sorted([bond_to_label(bond) for bond in atom1.GetBonds()])
    bonds2 = sorted([bond_to_label(bond) for bond in atom2.GetBonds()])
    if bonds1 != bonds2: return True

    # # Check neighbors too (already taken care of with previous lines)
    # neighbors1 = sorted([atom.GetAtomicNum() for atom in atom1.GetNeighbors()])
    # neighbors2 = sorted([atom.GetAtomicNum() for atom in atom2.GetNeighbors()])
    # if neighbors1 != neighbors2: return True

    # print('bonds1: {}'.format(bonds1))
    # print('bonds2: {}'.format(bonds2))

    return False


def find_map_num(mol, mapnum):
    return [(a.GetIdx(), a) for a in mol.GetAtoms() if a.HasProp('molAtomMapNumber')
            and a.GetProp('molAtomMapNumber') == str(mapnum)][0]


def get_tetrahedral_atoms(reactants, products):
    tetrahedral_atoms = []
    for reactant in reactants:
        for ar in reactant.GetAtoms():
            if not ar.HasProp('molAtomMapNumber'):
                continue
            atom_tag = ar.GetProp('molAtomMapNumber')
            ir = ar.GetIdx()
            for product in products:
                try:
                    (ip, ap) = find_map_num(product, atom_tag)
                    from rdkit.Chem.rdchem import ChiralType
                    if ar.GetChiralTag() != ChiralType.CHI_UNSPECIFIED or \
                            ap.GetChiralTag() != ChiralType.CHI_UNSPECIFIED:
                        tetrahedral_atoms.append((atom_tag, ar, ap))
                except IndexError:
                    pass
    return tetrahedral_atoms


def set_isotope_to_equal_mapnum(mol):
    for a in mol.GetAtoms():
        if a.HasProp('molAtomMapNumber'):
            a.SetIsotope(int(a.GetProp('molAtomMapNumber')))


def get_frag_around_tetrahedral_center(mol, idx):
    '''Builds a MolFragment using neighbors of a tetrahedral atom,
    where the molecule has already been updated to include isotopes'''
    ids_to_include = [idx]
    for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
        ids_to_include.append(neighbor.GetIdx())
    symbols = ['[{}{}]'.format(a.GetIsotope(), a.GetSymbol()) if a.GetIsotope() != 0 \
                   else '[#{}]'.format(a.GetAtomicNum()) for a in mol.GetAtoms()]
    return Chem.MolFragmentToSmiles(mol, ids_to_include, isomericSmiles=True,
                                    atomSymbols=symbols, allBondsExplicit=True,
                                    allHsExplicit=True)


def check_tetrahedral_centers_equivalent(atom1, atom2):
    '''Checks to see if tetrahedral centers are equivalent in
    chirality, ignoring the ChiralTag. Owning molecules of the
    input atoms must have been Isotope-mapped'''
    atom1_frag = get_frag_around_tetrahedral_center(atom1.GetOwningMol(), atom1.GetIdx())
    atom1_neighborhood = Chem.MolFromSmiles(atom1_frag, sanitize=False)
    for matched_ids in atom2.GetOwningMol().GetSubstructMatches(atom1_neighborhood, useChirality=True):
        if atom2.GetIdx() in matched_ids:
            return True
    return False


def clear_isotope(mol):
    [a.SetIsotope(0) for a in mol.GetAtoms()]


def get_changed_atoms(reactants, products):
    '''Looks at mapped atoms in a reaction and determines which ones changed'''

    err = 0
    prod_atoms, prod_atom_tags = get_tagged_atoms_from_mols(products)

    if v: print('Products contain {} tagged atoms'.format(len(prod_atoms)))
    if v: print('Products contain {} unique atom numbers'.format(len(set(prod_atom_tags))))

    reac_atoms, reac_atom_tags = get_tagged_atoms_from_mols(reactants)
    if len(set(prod_atom_tags)) != len(set(reac_atom_tags)):
        if v: print('warning: different atom tags appear in reactants and products')
        # err = 1 # okay for Reaxys, since Reaxys creates mass
    if len(prod_atoms) != len(reac_atoms):
        if v: print('warning: total number of tagged atoms differ, stoichometry != 1?')
        # err = 1

    # Find differences
    changed_atoms = []
    changed_atom_tags = []
    # print(reac_atom_tags)
    # print(prod_atom_tags)

    # Product atoms that are different from reactant atom equivalent
    for i, prod_tag in enumerate(prod_atom_tags):

        for j, reac_tag in enumerate(reac_atom_tags):
            if reac_tag != prod_tag: continue
            if reac_tag not in changed_atom_tags:  # don't bother comparing if we know this atom changes
                # If atom changed, add
                if atoms_are_different(prod_atoms[i], reac_atoms[j]):
                    changed_atoms.append(reac_atoms[j])
                    changed_atom_tags.append(reac_tag)
                    break
                # If reac_tag appears multiple times, add (need for stoichometry > 1)
                if prod_atom_tags.count(reac_tag) > 1:
                    changed_atoms.append(reac_atoms[j])
                    changed_atom_tags.append(reac_tag)
                    break

    # Reactant atoms that do not appear in product (tagged leaving groups)
    for j, reac_tag in enumerate(reac_atom_tags):
        if reac_tag not in changed_atom_tags:
            if reac_tag not in prod_atom_tags:
                changed_atoms.append(reac_atoms[j])
                changed_atom_tags.append(reac_tag)

    # Changed CHIRALITY atoms (just tetra for now...)
    tetra_atoms = get_tetrahedral_atoms(reactants, products)
    if v:
        print('Found {} atom-mapped tetrahedral atoms that have chirality specified at least partially'.format(
            len(tetra_atoms)))
    [set_isotope_to_equal_mapnum(reactant) for reactant in reactants]
    [set_isotope_to_equal_mapnum(product) for product in products]
    for (atom_tag, ar, ap) in tetra_atoms:
        if v:
            print('For atom tag {}'.format(atom_tag))
            print('    reactant: {}'.format(ar.GetChiralTag()))
            print('    product:  {}'.format(ap.GetChiralTag()))
        if atom_tag in changed_atom_tags:
            if v:
                print('-> atoms have changed (by more than just chirality!)')
        else:
            from rdkit.Chem.rdchem import ChiralType
            unchanged = check_tetrahedral_centers_equivalent(ar, ap) and \
                        ChiralType.CHI_UNSPECIFIED not in [ar.GetChiralTag(), ap.GetChiralTag()]
            if unchanged:
                if v:
                    print('-> atoms confirmed to have same chirality, no change')
            else:
                if v:
                    print('-> atom changed chirality!!')
                # Make sure chiral change is next to the reaction center and not
                # a random specifidation (must be CONNECTED to a changed atom)
                tetra_adj_to_rxn = False
                for neighbor in ap.GetNeighbors():
                    if neighbor.HasProp('molAtomMapNumber'):
                        if neighbor.GetProp('molAtomMapNumber') in changed_atom_tags:
                            tetra_adj_to_rxn = True
                            break
                if tetra_adj_to_rxn:
                    if v:
                        print('-> atom adj to reaction center, now included')
                    changed_atom_tags.append(atom_tag)
                    changed_atoms.append(ar)
                else:
                    if v:
                        print('-> adj far from reaction center, not including')
    [clear_isotope(reactant) for reactant in reactants]
    [clear_isotope(product) for product in products]

    if v:
        print('{} tagged atoms in reactants change 1-atom properties'.format(len(changed_atom_tags)))
        for smarts in [atom.GetSmarts() for atom in changed_atoms]:
            print('  {}'.format(smarts))

    return changed_atoms, changed_atom_tags, err


def get_changed_atom_map_from_rxn_smiles(reaction_smiles):
    '''Function to process one doc'''

    try:
        # Unpack
        if '[2H]' in reaction_smiles:
            # stupid, specific deuterated case makes RemoveHs not remove 2Hs
            reaction_smiles = re.sub('\[2H\]', r'[H]', reaction_smiles)

        reactants = mols_from_smiles_list(reaction_smiles.split('>>')[0].split('.'))
        products = mols_from_smiles_list(reaction_smiles.split('>>')[1].split('.'))
        if None in reactants: return
        if None in products: return
        for i in range(len(reactants)):
            reactants[i] = AllChem.RemoveHs(reactants[i])  # *might* not be safe
        for i in range(len(products)):
            products[i] = AllChem.RemoveHs(products[i])  # *might* not be safe
        [Chem.SanitizeMol(mol) for mol in reactants + products]  # redundant w/ RemoveHs
        [mol.UpdatePropertyCache() for mol in reactants + products]
    except Exception as e:
        # can't sanitize -> skip
        print(e)
        print('Could not load SMILES or sanitize')
        return

    try:
        ###
        ### Check product atom mapping to see if reagent contributes
        ###

        are_unmapped_product_atoms = False
        extra_reactant_fragment = ''
        for product in products:
            if sum([a.HasProp('molAtomMapNumber') for a in product.GetAtoms()]) < len(product.GetAtoms()):
                print('!!!! Not all product atoms have atom mapping')
                print(reaction_smiles)
                are_unmapped_product_atoms = True
        if are_unmapped_product_atoms:  # add fragment to template

            total_partialmapped += 1
            for product in products:
                # Get unmapped atoms
                unmapped_ids = [
                    a.GetIdx() for a in product.GetAtoms() if not a.HasProp('molAtomMapNumber')
                ]
                if len(unmapped_ids) > MAXIMUM_NUMBER_UNMAPPED_PRODUCT_ATOMS:
                    # Skip this example - too many unmapped!
                    return
                # Define new atom symbols for fragment with atom maps, generalizing fully
                atom_symbols = ['[{}]'.format(a.GetSymbol()) for a in product.GetAtoms()]
                # And bond symbols...
                bond_symbols = ['~' for b in product.GetBonds()]
                if unmapped_ids:
                    extra_reactant_fragment += \
                        AllChem.MolFragmentToSmiles(product, unmapped_ids,
                                                    allHsExplicit=False, isomericSmiles=USE_STEREOCHEMISTRY,
                                                    atomSymbols=atom_symbols, bondSymbols=bond_symbols) + '.'
            if extra_reactant_fragment:
                extra_reactant_fragment = extra_reactant_fragment[:-1]
                if v: print('    extra reactant fragment: {}'.format(extra_reactant_fragment))

            # Consolidate repeated fragments (stoichometry)
            extra_reactant_fragment = '.'.join(sorted(list(set(extra_reactant_fragment.split('.')))))
            # fragmatch = Chem.MolFromSmarts(extra_reactant_fragment) # no parentheses

        ###
        ### Do RX-level processing
        ###

        if v: print(reaction_smiles)
        if None in reactants + products:
            print('Could not parse all molecules in reaction, skipping')
            return

        # Calculate changed atoms
        changed_atoms, changed_atom_tags, err = get_changed_atoms(reactants, products)
        if err:
            print('Could not get changed atoms')
            return
        if not changed_atom_tags:
            print('No atoms changed?')
            return

        return changed_atom_tags

    except KeyboardInterrupt:
        print('Interrupted')
        raise KeyboardInterrupt

    except Exception as e:
        print(e)
        if v:
            print('skipping')
            # raw_input('Enter anything to continue')
        return


class get_split_bond_atom:
    def __init__(self, rxn_smiles):
        self.rxn_smiles = rxn_smiles

    def get_changed_map(self):
        rxn_smiles = self.rxn_smiles
        changed_map_all = get_changed_atom_map_from_rxn_smiles(rxn_smiles)
        changed_map_dic = {}
        changed_map_dic['pds'] = changed_map_all
        rts = rxn_smiles.split('>>')[0].split('.')
        for i, rt in enumerate(rts):
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
        for i, rt in enumerate(rts):
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
        return pds_nei_dic, rts_nei_dic

    def get_nei_diff(self):
        pds_nei_dic, rts_nei_dic = self.get_neighbor_map()
        split_bond_atom_map = []
        for map_ in pds_nei_dic.keys():
            if set(pds_nei_dic[map_]) != set(rts_nei_dic[map_]):
                #                 if len(list(set(pds_nei_dic[map_]))) < len(list(set(rts_nei_dic[map_]))):
                split_bond_atom_map.append(map_)
        return split_bond_atom_map

    def is_self(self):
        pds_nei_dic, rts_nei_dic = self.get_neighbor_map()
        set_pds_nei_dic = {}
        for key in pds_nei_dic.keys():
            set_pds_nei_dic[key] = set(pds_nei_dic[key])
        set_rts_nei_dic = {}
        for key in rts_nei_dic.keys():
            set_rts_nei_dic[key] = set(rts_nei_dic[key])
        if set_pds_nei_dic == set_rts_nei_dic:
            return True
        else:
            return False


def get_rxn_position_info_pd(rxn_smiles):
    test_rxn = rxn_smiles
    pp = get_split_bond_atom(test_rxn)

    rts, pds = test_rxn.split('>>')[0], test_rxn.split('>>')[-1]
    mol_pds = Chem.MolFromSmiles(pds)
    len_rts = len(rts.split('.'))
    if not pp.is_self():
        if len_rts == 2:
            changed_map_list = pp.get_nei_diff()  # 在这里表示断键两边的map
            if len(changed_map_list) == 2:  # 综合判断反应物数量和变化原子数量如果都为2那么标记1
                rxn_position_idx = []
                for atom in mol_pds.GetAtoms():
                    if atom.GetProp('molAtomMapNumber') in changed_map_list:
                        rxn_position_idx.append(atom.GetIdx())
                for atom in mol_pds.GetAtoms():
                    if atom.GetIdx() not in rxn_position_idx:
                        atom.ClearProp('molAtomMapNumber')
                    if atom.GetIdx() in rxn_position_idx:
                        atom.SetProp('molAtomMapNumber', '1')
                pds_smiles = Chem.MolToSmiles(mol_pds, canonical=False)
            elif len(changed_map_list) >= 3:  # 综合判断反应物数量和标记原子，如果反应物数量为2，变化原子数量大于等于3的化标记4
                rxn_position_idx = []
                for atom in mol_pds.GetAtoms():
                    if atom.GetProp('molAtomMapNumber') in changed_map_list:
                        rxn_position_idx.append(atom.GetIdx())
                for atom in mol_pds.GetAtoms():
                    if atom.GetIdx() not in rxn_position_idx:
                        atom.ClearProp('molAtomMapNumber')
                    if atom.GetIdx() in rxn_position_idx:
                        atom.SetProp('molAtomMapNumber', '4')
                pds_smiles = Chem.MolToSmiles(mol_pds, canonical=False)
            elif len(changed_map_list) == 1:  # 反应物数量2，变化原子1，标记3
                rxn_position_idx = []
                for atom in mol_pds.GetAtoms():
                    if atom.GetProp('molAtomMapNumber') in changed_map_list:
                        rxn_position_idx.append(atom.GetIdx())
                for atom in mol_pds.GetAtoms():
                    if atom.GetIdx() not in rxn_position_idx:
                        atom.ClearProp('molAtomMapNumber')
                    if atom.GetIdx() in rxn_position_idx:
                        atom.SetProp('molAtomMapNumber', '3')
                pds_smiles = Chem.MolToSmiles(mol_pds, canonical=False)
            else:
                print('err')
        elif len_rts == 1:
            changed_map_list = pp.get_nei_diff()
            if len(changed_map_list) == 1:
                rxn_position_idx = []
                for atom in mol_pds.GetAtoms():
                    if atom.GetProp('molAtomMapNumber') in changed_map_list:
                        rxn_position_idx.append(atom.GetIdx())
                for atom in mol_pds.GetAtoms():
                    if atom.GetIdx() not in rxn_position_idx:
                        atom.ClearProp('molAtomMapNumber')
                    if atom.GetIdx() in rxn_position_idx:
                        atom.SetProp('molAtomMapNumber', '3')
                pds_smiles = Chem.MolToSmiles(mol_pds, canonical=False)
            elif len(changed_map_list) == 2:
                rxn_position_idx = []
                for atom in mol_pds.GetAtoms():
                    if atom.GetProp('molAtomMapNumber') in changed_map_list:
                        rxn_position_idx.append(atom.GetIdx())
                for atom in mol_pds.GetAtoms():
                    if atom.GetIdx() not in rxn_position_idx:
                        atom.ClearProp('molAtomMapNumber')
                    if atom.GetIdx() in rxn_position_idx:
                        atom.SetProp('molAtomMapNumber', '2')
                pds_smiles = Chem.MolToSmiles(mol_pds, canonical=False)
            elif len(changed_map_list) >= 3:
                rxn_position_idx = []
                for atom in mol_pds.GetAtoms():
                    if atom.GetProp('molAtomMapNumber') in changed_map_list:
                        rxn_position_idx.append(atom.GetIdx())
                for atom in mol_pds.GetAtoms():
                    if atom.GetIdx() not in rxn_position_idx:
                        atom.ClearProp('molAtomMapNumber')
                    if atom.GetIdx() in rxn_position_idx:
                        atom.SetProp('molAtomMapNumber', '2')
                pds_smiles = Chem.MolToSmiles(mol_pds, canonical=False)
            else:
                print('err')
        elif len_rts >= 3:
            changed_map_list = pp.get_nei_diff()
            rxn_position_idx = []
            for atom in mol_pds.GetAtoms():
                if atom.GetProp('molAtomMapNumber') in changed_map_list:
                    rxn_position_idx.append(atom.GetIdx())
            for atom in mol_pds.GetAtoms():
                if atom.GetIdx() not in rxn_position_idx:
                    atom.ClearProp('molAtomMapNumber')
                if atom.GetIdx() in rxn_position_idx:
                    atom.SetProp('molAtomMapNumber', '4')
            pds_smiles = Chem.MolToSmiles(mol_pds, canonical=False)
    elif pp.is_self():  # 如果没有断键只有反应物内的重排则标签为:2
        rxn_position_idx = []
        changed_map_list = get_changed_atom_map_from_rxn_smiles(test_rxn)
        for atom in mol_pds.GetAtoms():
            if atom.GetProp('molAtomMapNumber') in changed_map_list:
                rxn_position_idx.append(atom.GetIdx())
        for atom in mol_pds.GetAtoms():
            if atom.GetIdx() not in rxn_position_idx:
                atom.ClearProp('molAtomMapNumber')
            if atom.GetIdx() in rxn_position_idx:
                atom.SetProp('molAtomMapNumber', '2')
        pds_smiles = Chem.MolToSmiles(mol_pds, canonical=False)
    else:
        print('err')
        return
    return pds_smiles


def transfor_mark(marked_prod, prod):
    mol_marked = Chem.MolFromSmiles(marked_prod)
    mol_canonical = Chem.MolFromSmiles(prod)
    sub_tuple = mol_marked.GetSubstructMatch(mol_canonical)
    assert len(sub_tuple) != 0
    #     c2m_index_dic = {i:sub_tuple[i] for i in range(mol_canonical.GetNumAtoms())}
    #     m2c_index_dic = {c2m_index_dic[i]:i for i in range(len(c2m_index_dic))}
    cindex2map_dic = {}
    for i, index in enumerate(sub_tuple):
        if mol_marked.GetAtomWithIdx(index).HasProp('molAtomMapNumber'):
            cindex2map_dic[i] = mol_marked.GetAtomWithIdx(index).GetProp('molAtomMapNumber')
    for i, atom in enumerate(mol_canonical.GetAtoms()):
        if i in cindex2map_dic:
            atom.SetProp('molAtomMapNumber', cindex2map_dic[i])
    return Chem.MolToSmiles(mol_canonical, canonical=False)


def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    import re
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)


def c2apbp(info_pds_smiles):
    '''
    拆解反应位点标1和4的分子，返回分子碎片（canonical）
    '''
    mol = Chem.MolFromSmiles(info_pds_smiles)
    maped_atom_index = []
    for atom in mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            if atom.GetProp('molAtomMapNumber') in ['1', '4']:
                maped_atom_index.append(atom.GetIdx())
    bonds_id = []
    maped_atom_index_copy = deepcopy(maped_atom_index)
    if len(maped_atom_index) == 1:
        return info_pds_smiles
    for x in maped_atom_index:
        for y in maped_atom_index_copy:
            if mol.GetBondBetweenAtoms(x, y) is not None:
                bonds_id.append(mol.GetBondBetweenAtoms(x, y).GetIdx())
    set_bonds_id = list(set(bonds_id))
    if len(set_bonds_id) == 0:
        return info_pds_smiles
    frags = Chem.FragmentOnBonds(mol, set_bonds_id)
    frags_smiles = Chem.MolToSmiles(frags, canonical=True)
    frags_smiles_sub = re.sub('\[([0-9]+)\*\]', '*', frags_smiles)
    if Chem.MolFromSmiles(frags_smiles_sub) is None:
        return Chem.MolToSmiles(Chem.MolFromSmarts(frags_smiles_sub), canonical=True)
    frags_end = Chem.MolToSmiles(Chem.MolFromSmiles(frags_smiles_sub), canonical=True)
    return frags_end


def c2apbp(info_pds_smiles):
    '''
    拆解反应位点标1和4的分子，返回分子碎片（canonical）
    '''
    mol = Chem.MolFromSmiles(info_pds_smiles)
    maped_atom_index = []
    for atom in mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            if atom.GetProp('molAtomMapNumber') in ['1', '4']:
                maped_atom_index.append(atom.GetIdx())
    bonds_id = []
    maped_atom_index_copy = deepcopy(maped_atom_index)
    if len(maped_atom_index) == 1:
        return info_pds_smiles
    for x in maped_atom_index:
        for y in maped_atom_index_copy:
            if mol.GetBondBetweenAtoms(x, y) is not None:
                bonds_id.append(mol.GetBondBetweenAtoms(x, y).GetIdx())
    set_bonds_id = list(set(bonds_id))
    if len(set_bonds_id) == 0:
        return info_pds_smiles
    frags = Chem.FragmentOnBonds(mol, set_bonds_id)
    frags_smiles = Chem.MolToSmiles(frags, canonical=True)
    frags_smiles_sub = re.sub('\[([0-9]+)\*\]', '*', frags_smiles)
    if Chem.MolFromSmiles(frags_smiles_sub) is None:
        return Chem.MolToSmiles(Chem.MolFromSmarts(frags_smiles_sub), canonical=True)
    frags_end = Chem.MolToSmiles(Chem.MolFromSmiles(frags_smiles_sub), canonical=True)
    return frags_end


def get_mark_apbp(canonical_pos_info_pd):
    # 传入的标记都需要只有一种
    mark = re.findall('\:([0-9]+)\]', canonical_pos_info_pd)[0]
    if mark == '1':
        split = True
        test_mol = Chem.MolFromSmiles(c2apbp(canonical_pos_info_pd))
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
        if len(nei1_pos_index_list) != 0:
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

def get_info_index(smiles):
    info_index_list = []
    mark = 0
    mol = Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            mark = atom.GetProp('molAtomMapNumber')
            info_index_list.append(atom.GetIdx())
    return list(set(info_index_list)), mark

def get_mark_ab(rxn):
    rts = rxn.split('>>')[0]
    split_bond_class = get_split_bond_atom(rxn)
    mol_rts = Chem.MolFromSmiles(rts)
    if not split_bond_class.is_self():
        rxn_map_pos_list = split_bond_class.get_nei_diff()
    elif split_bond_class.is_self():
        rxn_map_pos_list = get_changed_atom_map_from_rxn_smiles(rxn)
    rxn_pos_index = []
    for atom in mol_rts.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            if atom.GetProp('molAtomMapNumber') in rxn_map_pos_list:
                rxn_pos_index.append(atom.GetIdx())
    nei_index = []
    for atom in mol_rts.GetAtoms():
        if atom.GetIdx() in rxn_pos_index:
            atom.SetProp('molAtomMapNumber', '1')
            for nei in atom.GetNeighbors():
                if nei.HasProp('molAtomMapNumber'):
                    if nei.GetIdx() not in rxn_pos_index:
                        nei_index.append(nei.GetIdx())
                        nei.SetProp('molAtomMapNumber', '2')
    D_index = []
    for atom in mol_rts.GetAtoms():
        if not atom.HasProp('molAtomMapNumber'):
            D_index.append(atom.GetIdx())
            atom.SetProp('molAtomMapNumber', '1')
    for atom in mol_rts.GetAtoms():
        if atom.GetIdx() not in rxn_pos_index + nei_index + D_index:
            atom.SetProp('molAtomMapNumber', '3')
    ab_rt_smiles = Chem.MolToSmiles(mol_rts, canonical=True)
    return ab_rt_smiles


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


def smi_spliter(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    import re
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return tokens


def min_distance(src, tgt, mode='token'):
    def mode_select(str_, mode=mode):
        if mode == 'token':
            return smi_spliter(str_)
        elif mode == 'str':
            return str_

    ed_score = float('inf')
    src_mode = mode_select(src)
    tgt_split = tgt.split('.')
    if len(tgt_split) <= 5:  # 当枚举数据量太多的的话不进行序列优化
        tgt_perm = []
        for i in permutations(tgt_split, len(tgt_split)):
            tgt_perm.append('.'.join(i))
        for p in tgt_perm:
            p_mode = mode_select(p)
            check_score = editdistance(src_mode, p_mode)
            if check_score < ed_score:
                tgt_min_dis = p
                ed_score = check_score
            else:
                pass
        return tgt_min_dis
    else:  # 当枚举数据量太多的的话不进行序列优化
        return tgt


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

def is_number(s):
    try:
        float(s)
        return True
    except: return False

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

def MyGetAtom(smarts):
    '''
    获取atom smarts列表（因为rdkit自带的atom.GetSmarts()函数会使手性标记改变，编写这个函数是为了比对原子smarts，经过程序验证，本函数可以替代atom.GetSmarts(),不过返回的是atom smarts的列表。）
    :param smarts:
    :return:
    '''
    raw_list = split_smiles(smarts)
    for i in range(len(raw_list)):
        raw_list[i] = re.sub('[=,#,(,),.]', '', raw_list[i])
        if is_number(raw_list[i]):
            raw_list[i] = re.sub('[0-9]+', '', raw_list[i])
        if raw_list[i] is ':':
            raw_list[i] = raw_list[i].replace(':', '')
        if raw_list[i] is '-':
            raw_list[i] = raw_list[i].replace('-', '')
        if raw_list[i] is '/':
            raw_list[i] = raw_list[i].replace('/', '')
        if raw_list[i] is '\\':
            raw_list[i] = raw_list[i].replace('\\', '')
    atom_smarts_list = []
    for i in range(len(raw_list)):
        if raw_list[i] is not '':
            atom_smarts_list.append(raw_list[i])
    return atom_smarts_list

def get_info_num(atom_smarts):
    num = re.findall('\:([0-9]+)',atom_smarts)
    return num

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

def Execute_grammar_err(canonical_smiles, pre_smiles):
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
            except:
                flag = True
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
