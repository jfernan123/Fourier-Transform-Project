import numpy as np

def calculate_accuracy(a, b):
    same = np.equal(a, b)
    return np.sum(same) / same.size
