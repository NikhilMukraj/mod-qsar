from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from shapely.geometry import Point, Polygon
import numpy as np
import pandas as pd
import json
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow.keras import models
from smiles_tools import return_tokens
from smiles_tools import SmilesEnumerator
from bbb_gia import bbb_coords, gia_coords
from model_class import ClassifierModel, RegressionModel
from c_wrapper import seqOneHot
from ml_scorer import get_score
import string_ga
from functools import lru_cache
import importlib.util


GREEN = '\033[1;32m'
NC = '\033[0m'
RED = '\033[0;31m'

if len(sys.argv) < 3:
    print(f"{RED}Too few args...{NC}")
    sys.exit()

with open(sys.argv[1], 'r') as f:
    contents = json.load(f)

necessary_args = {
    'population_size': [int],
    'mating_pool_size': [int], 
    'generations': [int], 
    'mutation_rate': [float], 
    'seed': [int, type(None)], 
    'average_size': [int, float], 
    'size_stdev': [int, float], 
    'string_type': [str], 
    'scoring_function': [list], 
    'strict': [bool],
    'threads': [int],
    'augment': [list], 
    'max_len': [int, type(None)], 
    'max_score': [float], 
    'prune_population': [bool], 
    'target': [list], 
    'weight': [list],
    'file_name': [str],
    'vocab': [str],
}

optional_args = {
    "regression_min",
    "regression_max",
}

for i in contents.keys():
    if i not in necessary_args and i not in optional_args:
        print(f'{RED}Unknown argument: {i}{NC}')
        sys.exit(1)

for i in necessary_args.keys():
    if i not in contents:
        print(f'{RED}Missing argument: {i}{NC}')
        sys.exit(1)

for key, value in contents.items():
    if key not in optional_args and type(value) not in necessary_args[key]:
        print(f'{RED}Type mismatch in argument, expected type in {necessary_args[key]} at "{key}" argument but got {type(value)}{NC}')
        sys.exit(1)

if contents['population_size'] <= 0:
    print(f'{RED}"population_size" argument must be greater than 0{NC}')
    sys.exit(1)

if contents['mating_pool_size'] <= 0:
    print(f'{RED}"mating_pool_size" argument must be greater than 0{NC}')
    sys.exit(1)

if contents['generations'] <= 0:
    print(f'{RED}"generations" argument must be greater than 0{NC}')
    sys.exit(1)

if type(contents['augment'][0]) != bool:
    print(f'{RED}Type mismatch in argument, expected type {bool} at "augment" argument at first index but got {type(contents["augment"][0])}{NC}')
    sys.exit(1)

if contents['augment'][0] and len(contents['augment']) != 2:
    print(f'{RED}Expected second item specifying number of augmentations{NC}')
    sys.exit(1)

if len(contents['augment']) == 2 and type(contents['augment'][1]) != int:
    print(f'{RED}Type mismatch in argument, expected type {int} at "augment" argument at second but got {type(contents["augment"][1])}{NC}')
    sys.exit(1)

if contents['augment'][0] and contents['augment'][1] < 1:
    print(f'{RED}Expected integer greater than 0 but got {contents["augment"][1]}{NC}')
    sys.exit(1)

if any(type(i) != str for i in contents['scoring_function']):
    print(f'{RED}Type mistmatch in argument, expected type {str} in "scoring_function"{NC}')
    sys.exit(1)

def remove_next_dups(lst):
    i = 0
    while i < len(lst) - 1:
        if lst[i] == lst[i+1]:
            lst.pop(i+1)
        else:
            i += 1
    return lst

if len(dups := remove_next_dups(['.h5' in i for i in contents['scoring_function']])) > 2 \
    or (len(dups) <= 2 and dups[0] == 0 and dups[-1] == 1):
    print(f'{RED}"scoring_function" argument must specify models before all other scoring functions{NC}')
    sys.exit(1)

if len(dups := remove_next_dups([':' in i for i in contents['scoring_function']])) > 2 \
    or (len(dups) <= 2 and dups[0] == 1 and dups[-1] == 0):
    print(f'{RED}"scoring_function" argument must specify custom functions after all other scoring functions{NC}')
    sys.exit(1)

if contents['threads'] < 1: 
    print(f'{RED}Amount of threads must be 1 or more{NC}')
    sys.exit(1)

if not os.path.isfile(contents['file_name']):
    print(f'{RED}\"{contents["file_name"]}\" in "file_name" argument is not a valid file{NC}')
    sys.exit(1)

if not os.path.isfile(contents['vocab']):
    print(f'{RED}\"{contents["vocab"]}\" in "vocab" argument is not a valid file{NC}')
    sys.exit(1)

strict = contents['strict']

# applies qed drug likeness
def get_qed(molecule):
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

# applies lipinski's rule of 5 drug likeness
def get_lipinski(molecule):
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

# applies lipisnki's rule of 5 with weight requirements and 
# checks how synthesizable it is 
def get_custom_lipinski(molecule):
    weightThreshold = 200 >= Descriptors.ExactMolWt(molecule)
    saThreshold = string_ga.calculateScore(molecule) > 3

    if weightThreshold or saThreshold:
        return 0
    else: 
        return get_lipinski(molecule)

# removes larger rings
def get_ghose(molecule):
    ghose = 0

    molecular_weight = Descriptors.ExactMolWt(molecule)
    logp = Descriptors.MolLogP(molecule)
    number_of_atoms = Chem.rdchem.Mol.GetNumAtoms(molecule)
    molar_refractivity = Chem.Crippen.MolMR(molecule)

    if 160 <= molecular_weight <= 480:
        ghose += .25
    if -.4 <= logp <= 5.6:
        ghose += .25
    if 20 <= number_of_atoms <= 70:
        ghose += .25
    if 40 <= molar_refractivity <= 130:
        ghose += .25

    return ghose

# removes larger rings
def limit_rings(molecule):
    ring_lens = [len(ring) for ring in molecule.GetRingInfo().AtomRings() 
                if molecule.GetAtomWithIdx(ring[0]).GetSymbol() == 'C']
    for i in ring_lens:
        if i > 6:
            return 0
    
    return 1

params = FilterCatalogParams()
params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
catalog = FilterCatalog(params)

# helps with sensitivity
def get_pains(molecule):
    isPAINS = catalog.GetFirstMatch(molecule)
    return int(isPAINS is None)

# https://github.com/bfmilne/pyBOILEDegg/tree/main

gia = pd.DataFrame(gia_coords, columns=['tpsa', 'wlogp'])
bbb = pd.DataFrame(bbb_coords, columns=['tpsa', 'wlogp'])

gia_ellipse = Polygon(gia_coords)
bbb_ellipse = Polygon(bbb_coords)

# tests if passes blood brain barrier 
def bbb_permeable(molecule):
    tpsa = Descriptors.TPSA(molecule, includeSandP=True)
    wlogp = Descriptors.MolLogP(molecule)

    mol_coords = Point(tpsa, wlogp)

    if mol_coords.within(bbb_ellipse):
        return 1
    else:
        return 0

# tests how well intestinal tract absorbs it
def gastro_absorption(molecule):
    tpsa = Descriptors.TPSA(molecule, includeSandP=True)
    wlogp = Descriptors.MolLogP(molecule)

    mol_coords = Point(tpsa, wlogp)

    if mol_coords.within(gia_ellipse):
        return 1
    else:
        return 0

def module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def get_function(file_path, function_name):
    _, filename = os.path.split(file_path)
    module_name, _ = os.path.splitext(filename)
    module = module_from_file(module_name, file_path)

    return getattr(module, function_name)

def generate_function(func):
    return lambda x: float(func(x))

# mapping for drug metrics
drug_likeness_parser = {
    'lipinski': get_lipinski, 
    'custom_lipinski': get_custom_lipinski,
    'qed': get_qed, 
    'ghose': get_ghose,
    'pains': get_pains,
    'limit_rings': limit_rings,
    'bbb_permeable': bbb_permeable,
    'gastro_absorption': gastro_absorption,
}

drug_likeness_metric = [drug_likeness_parser[i] for i in contents['scoring_function'] 
                        if '.h5' not in i and ':' not in i]
custom_funcs = [generate_function(get_function(*i.split(':'))) 
                for i in contents['scoring_function'] if ':' in i]
drug_likeness_metric.extend(custom_funcs)

vocab = pd.read_csv(contents['vocab'])['tokens'].to_list()
tokenizer = {i : n for n, i in enumerate(vocab)}

def is_regression_model(string):
    return os.path.basename(string).startswith('regr')

potential_models = [i for i in contents['scoring_function'] if '.h5' in i]

if any(is_regression_model(i) for i in potential_models):
    regr_min_used = 'regression_min' in contents
    regr_max_used = 'regression_max' in contents
    if regr_max_used + regr_min_used < 2:
        print(f'{RED}"regression_min" and "regression_max" arguments must be used if using a regression model{NC}')
        sys.exit(1)

    if type(contents['regression_max']) not in [float, int] or type(contents['regression_min']) not in [float, int]:
        print(f'{RED}"regression_min" and "regression_max" arguments must be numeric{NC}')
        sys.exit(1)

    if contents['regression_min'] >= contents['regression_max']:
        print(f'{RED}"regression_min" must be lower than "regression_max"{NC}')
        sys.exit(1)

    regression_min = contents['regression_min']
    regression_max = contents['regression_max']

if len(potential_models) > 0:
    models_array = [ClassifierModel(i) if not is_regression_model(i) else RegressionModel(i, regression_min, regression_max)
                    for i in potential_models]
    print('Compiling models...')
    [model.compile() for model in models_array]
    if contents['max_len']:
        max_len = contents['max_len']
    elif len(set([i.model.layers[0].output_shape[1] for i in models_array])) != 1:
        print(f'{RED}All models must have same input shape{NC}')
        sys.exit(1)
    else:
        max_len = models_array[0].model.layers[0].output_shape[1]
    seq_shape = np.array([max_len, np.max([i+1 for i in tokenizer.values()])+1], dtype=np.int32)
    model_pred = True
else:
    model_pred = False

# potentially can leave this with a bunch of setup args (tokenizer, seq_shape, model_array)
def ensemble_predict(tokens):
    initial_seq = np.array([tokenizer[i]+1 for i in tokens])
    full_seq = np.hstack([np.zeros(max_len-len(initial_seq)), initial_seq])
    full_seq = seqOneHot(np.array(full_seq, dtype=np.int32), seq_shape).reshape(1, *seq_shape)
    
    # return np.hstack([i.predict(full_seq, verbose=0) for i in models_array])
    return np.hstack([i.predict(full_seq) for i in models_array])
    # encapsulate regression models in class to use predict on it and automatically do
    # clipping and normalization

def augment_smiles(string, n):
    sme = SmilesEnumerator()
    output = []
    for i in range(n):
        output.append(sme.randomize_smiles(string))
    
    return output

def get_augs(string, n):
    strings = [string] + augment_smiles(string, n)
    tokens_array = [return_tokens(i, vocab)[0] for i in strings]

    full_seqs = []
    for tokens in tokens_array:
        if any([i not in vocab for i in tokens]) or len(tokens) > max_len:
            continue
        
        initial_seq = np.array([tokenizer[n]+1 for n in tokens])
        full_seq = np.hstack([np.zeros(max_len-len(initial_seq)), initial_seq])
        full_seq = seqOneHot(np.array(full_seq, dtype=np.int32), seq_shape).reshape(1, *seq_shape)
        full_seqs.append(full_seq[0])

        ####################################################################################
        # may want to reformat to while loop to see if we can still reach n many compounds #
        ####################################################################################

    return np.array(full_seqs)

def strict_weight_req(molecule):
    return not (200 <= Descriptors.ExactMolWt(molecule) <= 500)

# def strict_pred_req(pred, target, length):
#     return [round(i) for i in pred[:length * 2]] == target[:length * 2]

@lru_cache(maxsize=256)
def no_model_scoring(string, target):
    _, isNotValidToken = return_tokens(string, vocab)

    if isNotValidToken:
        return -100
    else:
        molecule = Chem.MolFromSmiles(string)
        likeness_score = np.array(np.hstack([metric(molecule) for metric in drug_likeness_metric]))
        return -1 * get_score(weight * likeness_score, np.array(target))

@lru_cache(maxsize=256)
def model_scoring(string, scoring_args):
    target, aug, num_of_augments = scoring_args

    tokens, isNotValidToken = return_tokens(string, vocab)
    if isNotValidToken or len(tokens) > max_len:
        return -100

    molecule = Chem.MolFromSmiles(string)
    likeness_score = np.array(np.hstack([metric(molecule) for metric in drug_likeness_metric]))
    
    if not aug:
        pred = ensemble_predict(tokens)[0]
        return -1 * get_score(weight * np.hstack([pred, likeness_score]), np.array(target))
    else:
        augs = get_augs(string, num_of_augments)
        pred = np.hstack([i.predict(augs) for i in models_array]).sum(axis=0) / len(augs)

        if strict and strict_weight_req(molecule):
            return -20  
        else:
            return -1 * get_score(weight * np.hstack([pred, likeness_score]), np.array(target)) 

scoring_args = contents['target']

if len(scoring_args) != sum(i.model.layers[-1].output_shape[1] for i in models_array) + len(drug_likeness_metric):
    print(f'{RED}Target arugment does not match scoring functions{NC}')
    sys.exit(1)
if [i for i in scoring_args if not 0 <= i <= 1]:
    print(f"{RED}Target must be between 0 and 1{NC}")
    sys.exit(1)

if len(contents['weight']) != len(scoring_args):
    print(f'{RED}Weight arugment does not match target{NC}')
    sys.exit(1)
if any([type(i) not in [int, float] for i in contents['weight']]):
    print(f'{RED}Weight arugment contains non-numeric, expected numeric value{NC}')
    sys.exit(1)
    
weight = np.array(contents['weight'])

if model_pred:
    scoring_args = [scoring_args]
    if contents['augment'][0]:
        scoring_args.append(contents['augment'][0])
        scoring_args.append(contents['augment'][1])
    else:
        scoring_args.append(False)
        scoring_args.append(0)
    scoring_args[0] = tuple(scoring_args[0])

    scoring_function = model_scoring
else:
    scoring_function = no_model_scoring

scoring_args = tuple(scoring_args)

string_ga.co.average_size = contents['average_size']
string_ga.co.size_stdev = contents['size_stdev']
string_ga.co.string_type = contents['string_type']

print('Starting GA...')

(scores, population, high_scores, score_hist) = string_ga.GA([contents['population_size'], contents['file_name'], 
                                       scoring_function, contents['generations'],
                                       contents['mating_pool_size'], contents['mutation_rate'], 
                                       scoring_args, contents['max_score'],
                                       contents['prune_population'], contents['seed'], contents['threads']])

def sanitize_string(string):
    return Chem.MolToSmiles(Chem.MolFromSmiles(string, sanitize=True))

final_result_df = pd.DataFrame(zip(scores, [sanitize_string(i) for i in population]), columns=['score', 'string'])
final_result_df.sort_values(by='score', ascending=False)
final_result_df.to_csv(sys.argv[2], index=False)

if len(sys.argv) == 4:
    score_df = pd.DataFrame(score_hist)
    score_df.to_csv(sys.argv[3], index=False)

print(f'{GREEN}Wrote molecules to file{NC}')
