'''
Written Jan H. Jensen 2019 
Requires import of string_crossover and string_mutate in GB_GA
'''

from rdkit import Chem
import numpy as np
import time

import string_scoring_functions as sc
import string_crossover as co
import string_GA as ga 
import sys
from multiprocessing import Pool

n_tries = 2
population_size = 20 
mating_pool_size = 20
generations = 20
mutation_rate = 0.01
co.average_size = 39.15
co.size_stdev = 3.50
co.string_type = 'DeepSMILES'
scoring_function = sc.logP_score
max_score = 9999.
scoring_args = []
n_cpus = 2


file_name = sys.argv[1]

print('population_size', population_size)
print('mating_pool_size', mating_pool_size)
print('generations', generations)
print('mutation_rate', mutation_rate)
print('max_score', max_score)
print('average_size/size_stdev', co.average_size, co.size_stdev)
print('initial pool', file_name)
print('number of tries', n_tries)
print('number of CPUs', n_cpus)
print('')

results = []
size = []
t0 = time.time()
all_scores = []
generations_list = []
args = n_tries*[[population_size, file_name,scoring_function,generations,mating_pool_size,mutation_rate,scoring_args, max_score]]
with Pool(n_cpus) as pool:
    output = pool.map(ga.GA, args)

for i in range(n_tries):     
    #(scores, population) = ga.GA([population_size, file_name,scoring_function,generations,mating_pool_size,mutation_rate,scoring_args])
    (scores, population, generation) = output[i]
    all_scores.append(scores)
    print(f'{i} {scores[0]:.2f} {co.string2smiles(population[0])} {generation}')
    results.append(scores[0])
    generations_list.append(generation)

t1 = time.time()
print('')
print(f'max score {max(results):.2f}, mean {np.array(results).mean():.2f} +/- {np.array(results).std():.2f}')
print(f'mean generations {np.array(generations_list).mean():.2f} +/- {np.array(generations_list).std():.2f}')
print(f'time {(t1-t0)/60.0:.2f} minutes')
#print(max(size),np.array(size).mean(),np.array(size).std())

#print(all_scores)
