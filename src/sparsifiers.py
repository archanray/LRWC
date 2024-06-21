import numpy as np
from src.utils import sample_index

def modifiedBKKS21(data, size=10):
    # init sparse matrix
    sparse_data = np.zeros_like(data)
    
    #ell_1 norms
    ell_one_row_norms = np.linalg.norm(data, ord=1, axis=1)
    ell_one_row_norms_squared =  ell_one_row_norms ** 2
    
    # pij is the matrix of probabilities
    # assign values correctly
    p1 = np.abs(data) / np.sum(np.abs(data))
    p1 /= np.sum(p1)
    p2 = (np.abs(data) / ell_one_row_norms[:, None]) * \
        ((ell_one_row_norms_squared / np.sum(ell_one_row_norms_squared))[:, None])
    p2 /= np.sum(p2)
    pij = np.maximum(p1, p2)
    pij /= np.sum(pij)
    sampled_indices = sample_index(pij, size)
    
    sparse_data[sampled_indices] = data[sampled_indices]
    sparse_data[sampled_indices] = sparse_data[sampled_indices] / pij[sampled_indices] # elementwise divide
    
    return sparse_data