import numpy as np
from random import choices as randomChoices
from sklearn.utils.extmath import randomized_svd
import sys

def sort_abs_descending(x, type="values"):
    """
    Sort array by absolute value in descending order.
    """
    abs_x = np.abs(x)
    idx = np.argsort(-abs_x)
    if type == "values":
        return x[idx]
    else:
        return idx

def sort_descending(x, type="values"):
    """
    Sort array in descending order.
    """
    idx = np.argsort(-x)
    if type == "values":
        return x[idx]
    else:
        return idx
    
def sample_index(probs, size):
    """
    - true sampling with replacement of indices, returns indices, and weights.
    - sampling probabilities don't have to sum to one. instead, weights are 
    assigned before sampling.
    - weights \neq 1 iff the indices are sample multiple times
    """
    # sampling with replacement, so the list of indices might have duplicates
    
    # np.random.choices is about 5 times faster than random.choices
    # indices = randomChoices(np.arange(probs.size), weights=probs.ravel(), k=size)
    probs = probs / np.sum(probs)
    indices = np.random.choice(np.arange(probs.size), p=probs.ravel(), size=size, replace=True)
    # print("grabbed indices")
    # dict to find assigned weight -- takes O(size) time
    dict_indices = {}
    for i in range(len(indices)):
        if not dict_indices.get(indices[i], 0):
            dict_indices[indices[i]] = 1
        else:
            dict_indices[indices[i]] += 1
    indices = np.array(list(dict_indices.keys()))
    dupe_weights = np.array(list(dict_indices.values()))
    return np.unravel_index(indices, probs.shape), dupe_weights

def fast_spectral_norm(A):
    [U, sings, V] = randomized_svd(A, n_components=1, random_state=0)
    return sings[0]

# # test for sample_index
# size = 50
# A = np.random.random((20,10))
# Q = np.zeros_like(A)
# indices = sample_index(A, size)
# Q[indices] = A[indices]
# print(Q)