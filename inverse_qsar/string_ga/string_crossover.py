'''
Written by Emilie S. Henault and Jan H. Jensen 2019 
'''
from selfies import encoder, decoder
import deepsmiles
from rdkit import Chem
from rdkit.Chem import AllChem

import random
import numpy as np

from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

converter = deepsmiles.Converter(rings=True, branches=True)


def string_OK(string):
    mol = string2mol(string)
    if not mol:
        return False
    try:
        Chem.SanitizeMol(mol)
        test_mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        if test_mol == None:
            return None
        target_size = size_stdev*np.random.randn() + average_size  # parameters set in GA_mol
        if mol.GetNumAtoms() > 5 and mol.GetNumAtoms() < target_size:
            return True
        else:
            return False
    except:
        return False

def cut_point(parent):
    m = random.randint(0, len(parent) - 1)
    return m

def string2list(string):
    if string_type == 'SELFIES':
        return string.split('][')
    else:
        return list(string)

def mol2string(mol):
    smiles = Chem.MolToSmiles(mol)

    if string_type == 'SELFIES':
        return encoder(smiles).split('][')

    if string_type == 'DeepSMILES':
        string = converter.encode(smiles)
        return list(string)

    return list(smiles)

def list2string(list):
    if string_type == 'SELFIES':
        string = ']['.join(list)
    else:
        string = ''.join(list)

    return string

def smiles2string(smiles):
    if string_type == 'SMILES':
        string = smiles

    if string_type == 'SELFIES':
        try:
            string = encoder(smiles, PrintErrorMessage=False)
        except:
            return None

    if string_type == 'DeepSMILES':
        try:
            string = converter.encode(smiles)
        except deepsmiles.DecodeError as e:
            return None

    return string

def string2smiles(string):
    if string_type == 'SMILES':
        smiles = string

    if string_type == 'SELFIES':
        try:
            smiles = decoder(string, PrintErrorMessage=False)
        except:
            return None

    if string_type == 'DeepSMILES':
        try:
            smiles = converter.decode(string)
        except:
            return None

    return smiles

def string2mol(string):
    smiles = string2smiles(string)
    if smiles == None:
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except:
        return None

def crossover(parent_a, parent_b):
    # parent_a, parent_b = string2list(parent_a), string2list(parent_a)
    for _ in range(50):
        cut_point_a = cut_point(parent_a)
        cut_point_b = cut_point(parent_b)
        a1 = parent_a[0:cut_point_a]
        b2 = parent_b[cut_point_b:len(parent_b)]
        child_string = a1 + b2
        # print(child_string,Chem.MolToSmiles(child_mol),child_mol,co.mol_OK(child_mol))
        if string_OK(child_string):
            return child_string

    return None

if __name__ == "__main__":
    average_size = 39.15
    size_stdev = 3.50
    string_type = 'SELFIES'
    parent_a = 'CCCCCCCC'
    parent_b = 'OCCCCCCO'
    # parent_b = 'OCCCCCCc1ccccc1'
    child = crossover(parent_a, parent_b)
    print(child)
