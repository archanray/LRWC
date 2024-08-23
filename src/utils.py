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


def count_elements_for_threshold(row, T, d, sum_mode="ell_one"):
    """
    check per row total count of elements not exceeding the threshold
    """
    sorted_row = sorted(row)
    current_sum = 0
    count = 0
    for j in range(d-1,-1,-1):
        value = sorted_row[j]
        if sum_mode == "ell_one":
            if current_sum + value <= T:
                current_sum += value
                count += 1
        if sum_mode == "ell_two":
            if current_sum + value**2 <= T:
                current_sum += value**2
                count += 1
        else:
            break
    return count

def can_achieve_threshold(matrix, T, m, n, d, sum_mode="ell_one"):
    """
    see if for the given threshold we can limit the total number of elements to m
    """
    total_count = 0
    for i in range(n):
        total_count += count_elements_for_threshold(matrix[i,:], T, d, sum_mode)
        if total_count > m:
            return False
    return total_count <= m

def find_threshold(matrix, m, sum_mode="ell_one"):
    """
    general flow:
    1. find low = 0, sum of all large elements
    2. binary search over low and high and see if threshold can be achieved per row
    3. If yes, increase low, else decrease high
    """
    if sum_mode == "ell_one":
        low, high = 0, sum(max(row) for row in matrix)
    if sum_mode == "ell_two":
        low, high = 0, sum(max(row)**2 for row in matrix)
    best_threshold = low
    n,d = matrix.shape
    
    while low <= high:
        mid = (low + high) // 2
        if can_achieve_threshold(matrix, mid, m, n, d, sum_mode):
            best_threshold = mid
            low = mid + 1
        else:
            high = mid - 1
    
    return best_threshold

# # test for sample_index
# size = 50
# A = np.random.random((20,10))
# Q = np.zeros_like(A)
# indices = sample_index(A, size)
# Q[indices] = A[indices]
# print(Q)