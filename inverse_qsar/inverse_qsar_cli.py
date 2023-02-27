from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import py3Dmol
import numpy as np
import pandas as pd
import json
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow.keras import models
from smiles_tools import return_tokens
from c_wrapper import seqOneHot
from ml_scorer import get_score
import string_ga
from functools import lru_cache


if len(sys.argv) < 3:
    print("Too few args...")
    sys.exit()

with open(sys.argv[1], 'r') as f:
    contents = json.load(f)

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
    if 200 <= molecular_weight <= 500:
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

vocab = pd.read_csv('../preprocessor/vocab.csv')['tokens'].to_list()
tokenizer = {i : n for n, i in enumerate(vocab)}

potential_models = [i for i in contents["scoring_function"] if '.h5' in i]
if len(potential_models) > 0:
    models_array = [models.load_model(f'{os.getcwd()}//{i}') for i in potential_models]
    [model.compile() for model in models_array]
    max_len = 190
    seq_shape = np.array([max_len, np.max([i+1 for i in tokenizer.values()])+1], dtype=np.int32)
    model_pred = True
else:
    model_pred = False

drug_likeness_dict = {'lipinski': get_lipinski, 'qed': get_qed}
drug_likeness_to_use = [drug_likeness_dict[i] for i in contents['scoring_function'] if '.h5' not in i]

target = contents['target']
target = tuple(target)

@lru_cache(maxsize=256)
def no_model_scoring(string, target):
    raw_return = return_tokens(string, vocab)
    isNotValidToken = raw_return[1]
    if isNotValidToken:
        return -100
    else:
        likeness_score = np.array(np.hstack([func(string) for func in drug_likeness_to_use]))
        return -1 * get_score(likeness_score, np.array(target))

@lru_cache(maxsize=256)
def model_scoring(string, target):
    raw_return = return_tokens(string, vocab)
    isNotValidToken = raw_return[1]
    tokens = raw_return[0]
    if isNotValidToken:
        return -100
    else:
        likeness_score = np.array(np.hstack([func(string) for func in drug_likeness_to_use]))
        pred = ensemble_predict(tokens)[0]
        return -1 * get_score(np.hstack([pred, likeness_score]), np.array(target))

if model_pred:
    scoring_function = model_scoring
else:
    scoring_function = no_model_scoring

string_ga.co.average_size = contents['average_size']
string_ga.co.size_stdev = contents['size_stdev']
string_ga.co.string_type = contents['string_type']

print('Starting ga...')

(scores, population, high_scores, generation) = string_ga.GA([contents['population_size'], contents['file_name'], 
                                       scoring_function, contents['generations'],
                                       contents['mating_pool_size'], contents['mutation_rate'], 
                                       target, contents['max_score'],
                                       contents['prune_population'], contents['seed']])

final_result_df = pd.DataFrame(set(high_scores))
final_result_df.columns = ['score', 'string']
final_result_df.sort_values(by='score', ascending=False)
final_result_df.to_csv(sys.argv[2], index=False)

print('Wrote molecules to file')
