'''
Written by Emilie S. Henault and Jan H. Jensen 2019 
'''
from rdkit import Chem
from rdkit.Chem import AllChem

import random
import numpy as np

import string_ga.string_crossover as co

from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

def get_symbols():
    if co.string_type == 'SMILES':
        symbols = ['C', 'O', '(', '=', ')', '[C@@H]', '[C@H]', 'H', '1', 'N', '2', '3', 'F', 'S', 
                    'Cl', '#', '+', '-', '/', '4', 'B', 'Br', '\\', '5', 'I']
    
    if co.string_type == 'DeepSMILES':
        symbols = ['C', 'N', ')', 'S', '=', 'O', '6', '5', '9', 'B', 'Br', '[C@@H]', '[C@H]', 'H', 
                    '+', '%10', '%11', '%12', '%13', '%14', '%15', '%16', '%17', '1', '0', '2', 'l',
                    '3', 'F', '#', '7', 'I', '-', '/', '\\', '4', '8']
    
    if co.string_type == 'SELFIES':
        symbols = ['C', 'Branch1_2', 'epsilon', 'Branch1_3', '=C', 'O', '#N', '=O', 'N', 'Ring1', 
               'Branch1_1', 'F', '=N', '#C', 'C@@H', 'S', 'Branch2_2', 'Ring2', 'Branch2_3', 
               'Branch2_1', 'Cl', 'O-', 'C@H', 'NH+', 'C@', 'Br', '/C', '/O', 'NH3+', '=S', 'NH2+', 
               'C@@', '=N+', '=NH+', 'N+', '\\C', '\\O', '/N', '/S', '\\S', 'S@', '\\O-', 'N-', '/NH+', 
               'S@@', '=NH2+', '/O-', 'S-', '/S-', 'I', '\\N', '\\Cl', '=P', '/F', '/C@H', '=OH+', 
                '\\S-', '=S@@', '/C@@H', 'P', '=S@', '\\C@@H', '/S@', '/Cl', '=N-', '/N+', 'NH-', 
                '\\C@H', 'P@@H', 'P@@', '\\N-', 'Expl\\Ring1', '=P@@', '=PH2', '#N+', '\\NH+', 'P@', 
                'P+', '\\N+', 'Expl/Ring1', 'S+', '=O+', '/N-', 'CH2-', '=P@', '=SH+', 'CH-', '/Br', 
                '/C@@', '\\Br', '/C@', '/O+', '\\F', '=S+', 'PH+', '\\NH2+', 'PH', '/NH-', '\\S@', 'S@@+', 
                '/NH2+', '\\I']

    return symbols


def mutate(child,mutation_rate):
    if random.random() > mutation_rate:
        return child
    symbols = get_symbols()
    child = co.string2list(child)
    for i in range(50):
        random_number = random.random()
        mutated_gene = random.randint(0, len(child) - 1)
        random_symbol_number = random.randint(0, len(symbols)-1)
        new_child = list(child)
        random_number = random.random()
        new_child[mutated_gene] = symbols[random_symbol_number]
        #print(child_smiles,Chem.MolToSmiles(child_mol),child_mol,co.mol_OK(child_mol))
        new_child = co.list2string(new_child)
        if co.string_OK(new_child):
            return new_child

    return co.list2string(child)

if __name__ == "__main__":
    co.average_size = 39.15
    co.size_stdev = 3.50
    mutation_rate = 1.0
    co.string_type = 'SMILES'
    string = 'CCC(CCCC)C'
    child = mutate(string,mutation_rate)
    print(child)
