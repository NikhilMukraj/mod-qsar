from rdkit import Chem
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat

import string_ga.string_crossover as co
import string_ga.scoring_functions as sc

def logP_max(string,dummy):
    mol = co.string2mol(string)
    score = sc.logP_score(mol)
    return max(0,score)

def logP_target(string,args):
    mol = co.string2mol(string)
    score = sc.logP_target(mol,args)
    return score

def rediscovery(string,args):
    mol = co.string2mol(string)
    score = sc.rediscovery(mol,args)
    return score

def absorbance_target(string,args):
    mol = co.string2mol(string)
    score = sc.absorbance_target(mol,args)
    return score

def calculate_scores(population,function,scoring_args):
  scores = []
  for gene in population:
    score = function(gene,scoring_args)
    scores.append(score)

  return scores

def thread_calc_scores(population,function,scoring_args,max_workers):
    executor = ThreadPoolExecutor(max_workers=max_workers)
    return list(executor.map(function,population,repeat(scoring_args)))

if __name__ == "__main__":
    co.average_size = 39.15
    co.size_stdev = 3.50
    Celecoxib = 'O=S(=O)(c3ccc(n1nc(cc1c2ccc(cc2)C)C(F)(F)F)cc3)N'
    target = Chem.MolFromSmiles(Celecoxib)
    co.string_type = 'smiles'
    string = 'CCCCCCCC'
    score = logP_max(string,[])
    score = rediscovery(string,[target])
    print(score)
