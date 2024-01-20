import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow.keras.models import load_model
import tensorflow as tf
from sklearn.metrics import accuracy_score
from smiles_tools import return_tokens
from smiles_tools import SmilesEnumerator
from c_wrapper import seqOneHot
from tqdm import tqdm
import numpy as np
import random
import pandas as pd
import sys


model = load_model(sys.argv[1])
model.compile()

X = np.load(sys.argv[2])
Y = np.load(sys.argv[3])

if sys.argv[4].isdigit():
    sampling_amount = len(Y) // int(sys.argv[4])
    print(f'Sampling {sampling_amount}...')
    indicies = random.sample(range(Y.shape[0]), sampling_amount)

    X = np.array([X[i] for i in indicies])
    Y = np.array([Y[i] for i in indicies])

vocab = pd.read_csv(sys.argv[5])['tokens'].to_list()
tokenizer = {i : n for n, i in enumerate(vocab)}
reverse_tokenizer = {value: key for key, value in tokenizer.items()}
convert_back = lambda x: ''.join(reverse_tokenizer.get(np.argmax(i)-1, '') for i in x)

max_len = X.shape[1]
seq_shape = np.array([max_len, np.max([i+1 for i in tokenizer.values()])+1], dtype=np.int32)

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
        initial_seq = np.array([tokenizer[n]+1 for n in tokens])
        full_seq = np.hstack([np.zeros(max_len-len(initial_seq)), initial_seq])
        full_seq = seqOneHot(np.array(full_seq, dtype=np.int32), seq_shape).reshape(1, *seq_shape)
        full_seqs.append(full_seq[0])
        
    return np.array(full_seqs)

print('Calculating accuracies...')
preds = model.predict(X)
initial_acc = accuracy_score([np.argmax(i) for i in Y], [np.argmax(i) for i in preds])

testing_range = range(int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8]))
accs = {i : 0 for i in testing_range}
for index, num in enumerate(testing_range):
    aug_preds = []
    print(f'iteration: {num}')
    for i in tqdm(range(len(X))):
        strings = [convert_back(X[i])] + augment_smiles(convert_back(X[i]), num) 
        tokens_array = [return_tokens(string, vocab)[0] for string in strings]

        full_seqs = []
        for tokens in tokens_array:
            initial_seq = np.array([tokenizer[n]+1 for n in tokens])
            full_seq = np.hstack([np.zeros(max_len-len(initial_seq)), initial_seq])
            full_seq = seqOneHot(np.array(full_seq, dtype=np.int32), seq_shape).reshape(1, *seq_shape)
            full_seqs.append(full_seq[0])

        current_pred = model.predict(np.array(full_seqs), verbose=0)
        aug_preds.append(current_pred.sum(axis=0) / len(current_pred))
    
    accs[num] = sum([np.argmax(i[0]) == np.argmax(i[1]) for i in zip(aug_preds, Y)]) / len(Y)

pd.DataFrame({'n': [0] + list(accs.keys()), 
              'accuracy': [initial_acc] + list(accs.values())}).to_csv(sys.argv[9], index=False)
              