'''
Written by Jan H. Jensen 2018. 
Many subsequent changes inspired by https://github.com/BenevolentAI/guacamol_baselines/tree/master/graph_ga
'''

import string_ga.string_scoring_functions as sc
import string_ga.string_mutate as mu
import string_ga.string_crossover as co
import sys
import time
import random
from tqdm import tqdm
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops

from rdkit import rdBase
rdBase.DisableLog('rdApp.error')


def read_file(file_name):
    smiles_list = []
    with open(file_name, 'r') as file:
        for smiles in file:
            smiles = smiles.replace('\n', '')
            smiles_list.append(smiles)

    return smiles_list


def make_initial_population(population_size, file_name):
    smiles_list = read_file(file_name)
    population = []
    for _ in range(population_size):
        smiles = random.choice(smiles_list)
        mol = Chem.MolFromSmiles(smiles)
        Chem.Kekulize(mol, clearAromaticFlags=True)
        smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
        string = co.smiles2string(smiles)
        # print(smiles)
        if string:
            population.append(string)

    return population


def calculate_normalized_fitness(scores):
    sum_scores = sum(scores)
    normalized_fitness = [score/sum_scores for score in scores]

    return normalized_fitness


def make_mating_pool(population, fitness, mating_pool_size):
    mating_pool = []
    for i in range(mating_pool_size):
        mating_pool.append(np.random.choice(population, p=fitness))

    return mating_pool


def reproduce(mating_pool, population_size, mutation_rate):
    new_population = []
    while len(new_population) < population_size:
        parent_A = random.choice(mating_pool)
        parent_B = random.choice(mating_pool)
        new_child = co.crossover(parent_A, parent_B)
        if new_child != None:
            mutated_child = mu.mutate(new_child, mutation_rate)
            if mutated_child != None:
                # print(','.join([mutated_child,new_child,parent_A,parent_B]))
                new_population.append(mutated_child)

    return new_population


def sanitize(population, scores, population_size, prune_population):
    if prune_population:
        smiles_list = []
        population_tuples = []
        for score, string in zip(scores, population):
            canonical_smiles = Chem.MolToSmiles(co.string2mol(string))
            if canonical_smiles not in smiles_list:
                smiles_list.append(canonical_smiles)
                population_tuples.append((score, string))
    else:
        population_tuples = list(zip(scores, population))

    population_tuples = sorted(population_tuples, key=lambda x: x[0], reverse=True)[:population_size]
    new_population = [t[1] for t in population_tuples]
    new_scores = [t[0] for t in population_tuples]

    return new_population, new_scores


current_generation = {'gen': 0}


def GA(args):
    population_size, file_name, scoring_function, generations, mating_pool_size, mutation_rate, \
    scoring_args, max_score, prune_population, seed, threads = args

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    population = make_initial_population(population_size, file_name)
    # print(population)
    scores = sc.calculate_scores(population, scoring_function, scoring_args)
    # population, scores = sanitize(population, scores, population_size, False)
    high_scores = []
    # high_scores.append((scores[0],population[0]))
    fitness = calculate_normalized_fitness(scores)

    score_history = []

    for generation in tqdm(range(generations)):
        current_generation['gen'] += 1
        mating_pool = make_mating_pool(population, fitness, mating_pool_size)
        new_population = reproduce(mating_pool, population_size, mutation_rate)

        if threads == 1:
            new_scores = sc.calculate_scores(
                new_population, scoring_function, scoring_args)
        else:
            new_scores = sc.thread_calc_scores(
                new_population, scoring_function, scoring_args, threads)

        score_history.append(new_scores)

        population, scores = sanitize(
            population+new_population, scores+new_scores, population_size, prune_population)
        fitness = calculate_normalized_fitness(scores)
        high_scores.append((scores[0], population[0]))

        if scores[0] >= max_score:
            break

    return (scores, population, high_scores, score_history)


if __name__ == "__main__":
    Celecoxib = 'O=S(=O)(c3ccc(n1nc(cc1c2ccc(cc2)C)C(F)(F)F)cc3)N'
    target = Chem.MolFromSmiles(Celecoxib)
    population_size = 20
    mating_pool_size = 20
    generations = 20
    mutation_rate = 0.01
    seed = 1
    co.average_size = 39.15
    co.size_stdev = 3.50
    scoring_function = sc.rediscovery  # sc.logP_score
    # scoring_function = sc.logP_score
    max_score = 1.0  # 9999.
    prune_population = True
    scoring_args = [target]
    threads = 1

    co.string_type = 'SMILES'

    file_name = 'ZINC_first_1000.smi'

    (scores, population, high_scores, generation) = GA([population_size, file_name, scoring_function, generations,
                                                        mating_pool_size, mutation_rate, scoring_args, max_score,
                                                        prune_population, seed, threads])
    print('done')
    print(high_scores)
