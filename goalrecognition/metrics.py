import numpy as np


def entropy(x):
    return -np.sum(x * np.log(x))
