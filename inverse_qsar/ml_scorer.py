from sklearn.metrics import log_loss
import numpy as np
import pandas as pd
from smiles_tools import return_tokens
from c_wrapper import seqOneHot


def augmented_predict(model, string, n=5):
    ############ major problem ##################
    # [nh] is a token that idk what to do about #
    # pls find fix                              #
    #############################################
    # https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system
    
    # use julia func to convert and return seqs
    # integrate sequence encoding directly into that func
    raise NotImplementedError('not implemented yet')

def ensemble_predict(string, seq_shape, models_array, vocab=None, tokenizer=None, max_len=190):
    if vocab is None:
        vocab = pd.read_csv(f'{os.getcwd()}//vocab.csv')['tokens'].to_list()
    
    if tokenizer is None:
        tokenizer = {i : n for n, i in enumerate(vocab)}
        
    initial_seq = np.array([tokenizer[i]+1 for i in return_tokens(string, vocab)[0]])
    full_seq = np.hstack([np.zeros(max_len-len(initial_seq)), initial_seq])
    full_seq = seqOneHot(np.array(full_seq, dtype=np.int32), seq_shape).reshape(1, *seq_shape)
    
    return np.hstack([i.predict(full_seq, verbose=0) for i in models_array])

# def get_score(predicted, target):
#     # should maybe inverse, or remove reverse=True from sorted
#     return log_loss(predicted, target)

def get_score(predictions, targets):
    N = predictions.shape[0]
    ce = -np.sum(targets * np.log(predictions)) / N
    return ce