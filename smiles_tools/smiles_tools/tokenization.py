from SmilesPE.pretokenizer import atomwise_tokenizer
import pandas as pd
import os


def return_tokens(string, vocab=None):
    '''
    Return SMILES tokens of string and if they are in vocabulary
    '''
    tokens = atomwise_tokenizer(string)
    if vocab is None:
        try:
            possible_tokens = pd.read_csv(f'{os.getcwd()}\\vocab.csv').iloc[:, 0].to_list()
        except FileNotFoundError:
            possible_tokens = pd.read_csv(f'{os.getcwd()}//vocab.csv').iloc[:, 0].to_list()
    else:
        possible_tokens = vocab
    
    return tokens, any(i not in possible_tokens for i in tokens)