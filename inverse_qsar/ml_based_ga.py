from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import string_crossover as co
import string_mutate as mu
import string_scoring_functions as sc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow.keras import models
from ml_scorer import get_score, ensemble_predict
from string_GA import GA


models_array = [models.load_model(f'{os.getcwd()}//{i}') for i in os.listdir() if '.h5' in i]
for model in models_array:
    model.compile()
vocab = pd.read_csv('../preprocessor/vocab.csv')['tokens'].to_list()
tokenizer = {i : n for n, i in enumerate(vocab)}

def get_score_from_string(string, target):
    predicted = ensemble_predict(string, models_array, vocab=vocab, tokenizer=tokenizer)
    return get_score(predicted, target)

population_size = 20 
mating_pool_size = 20
generations = 20
mutation_rate = 0.01
seed = None
co.average_size = 39.15
co.size_stdev = 3.50
co.string_type = 'SMILES'
scoring_function = get_score_from_string 
max_score = 1.0 # 9999
prune_population = True
target = np.array([1, 0, 1, 0])
scoring_args = [target]

file_name = 'ZINC_first_1000.smi'

(scores, population, high_scores, generation) = GA([population_size, file_name, scoring_function, generations,
                                       mating_pool_size, mutation_rate, scoring_args, max_score,
                                       prune_population, seed])
print('done')    
print(high_scores[0])