import numpy as np


def get_score(y_pred, y_true):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    term0 = (1 - y_true) * np.log(1 - y_pred + 1e-7)
    term1 = y_true * np.log(y_pred + 1e-7)
    return -np.mean(term0 + term1, axis=0)
