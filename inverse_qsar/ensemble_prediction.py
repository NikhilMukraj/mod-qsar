from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import py3Dmol
import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow.keras import models
from smiles_tools import return_tokens
from c_wrapper import seqOneHot
from ml_scorer import get_score
import string_ga
from functools import lru_cache


def get_qed(string):
    molecule = Chem.MolFromSmiles(string)
    qed = 0
    
    molecular_weight = Descriptors.ExactMolWt(molecule)
    logp = Descriptors.MolLogP(molecule)
    h_bond_donor = Descriptors.NumHDonors(molecule)
    h_bond_acceptors = Descriptors.NumHAcceptors(molecule)
    rotatable_bonds = Descriptors.NumRotatableBonds(molecule)
    num_of_rings = Chem.rdMolDescriptors.CalcNumRings(molecule)

    if molecular_weight < 400:
        qed += 1/6
    if num_of_rings > 0:
        qed += 1/6
    if rotatable_bonds < 5:
        qed += 1/6
    if h_bond_donor <= 5:
        qed += 1/6
    if h_bond_acceptors <= 10:
        qed += 1/6
    if logp < 5:
        qed += 1/6
        
    return qed

def get_lipinski(string):
    molecule = Chem.MolFromSmiles(string)
    lipi = 0

    h_bond_donors = Lipinski.NumHDonors(molecule)
    h_bond_acceptors = Descriptors.NumHAcceptors(molecule)
    molecular_weight = Descriptors.ExactMolWt(molecule)
    logp = Descriptors.MolLogP(molecule)

    if h_bond_donors <= 5:
        lipi += .25
    if h_bond_acceptors <= 10:
        lipi += .25
    if molecular_weight <= 500:
        lipi += .25
    if logp <= 5:
        lipi += .25

    return lipi

# potentially can leave this with a bunch of setup args (tokenizer, seq_shape, model_array)
def ensemble_predict(tokens):
    initial_seq = np.array([tokenizer[i]+1 for i in tokens])
    full_seq = np.hstack([np.zeros(max_len-len(initial_seq)), initial_seq])
    full_seq = seqOneHot(np.array(full_seq, dtype=np.int32), seq_shape).reshape(1, *seq_shape)
    
    return np.hstack([i.predict(full_seq, verbose=0) for i in models_array])

# https://sulstice.medium.com/understanding-drug-likeness-filters-with-rdkit-and-exploring-the-withdrawn-database-ebd6b8b2921e
@lru_cache(maxsize=256)
def scoring(string, models_array):
    raw_return = return_tokens(string, vocab)
    isNotValidToken = raw_return[1]
    tokens = raw_return[0]
    if isNotValidToken:
        return -100
    else:
        return -1 * get_score(np.hstack([ensemble_predict(tokens)[0], np.array([get_lipinski(string)])]), np.array([1, 0, 1, 0, 1]))
        # return -1 * get_score(ensemble_predict(tokens), np.array([1, 0, 1, 0]))

if __name__ == '__main__':
    models_array = [models.load_model(f'{os.getcwd()}//{i}') for i in os.listdir() if '.h5' in i]
    [model.compile() for model in models_array]
    vocab = pd.read_csv('../preprocessor/vocab.csv')['tokens'].to_list()
    tokenizer = {i : n for n, i in enumerate(vocab)}
    max_len = 190
    seq_shape = np.array([max_len, np.max([i+1 for i in tokenizer.values()])+1], dtype=np.int32)

    population_size = 100 
    mating_pool_size = 100
    generations = 20
    mutation_rate = 0.05
    seed = None
    string_ga.co.average_size = 39.15
    string_ga.co.size_stdev = 3.50
    string_ga.co.string_type = 'SMILES'
    scoring_function = scoring
    max_score = 1.0 # 9999
    prune_population = True
    # target = np.array([1, 0, 1, 0])
    scoring_args = tuple(models_array)
    file_name = 'string_ga//ZINC_first_1000.smi'

    print('starting ga...')

    (scores, population, high_scores, generation) = string_ga.GA([population_size, file_name, scoring_function, generations,
                                        mating_pool_size, mutation_rate, scoring_args, max_score,
                                        prune_population, seed])

    final_result_df = pd.DataFrame(set(high_scores))
    final_result_df.columns = ['score', 'string']
    final_result_df.sort_values(by='score', ascending=False)
    final_result_df.to_csv(f'{os.getcwd()}//final_drugs.csv', index=False)

    print('wrote molecules to file')