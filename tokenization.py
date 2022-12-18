from SmilesPE.pretokenizer import atomwise_tokenizer
import pandas as pd
import os


def return_tokens(string):
    '''
    Return SMILES tokens of string and if they are in vocabulary
    '''
    tokens = atomwise_tokenizer(string)
    possible_tokens = pd.read_csv(f'{os.getcwd()}\\vocab.csv').iloc[:, 0].to_list()
    
    return tokens, any(i not in possible_tokens for i in tokens)