import numpy as np


def get_score(predictions, targets):
    N = predictions.shape[0]
    ce = -np.sum(targets * np.log(predictions)) / N
    if np.isnan(ce):
        return 0
    elif ce == np.inf:
        return 20
    else:
        return ce