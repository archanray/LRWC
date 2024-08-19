import numpy as np
import scipy.sparse
from src.utils import sample_index
import math
import scipy
from src.utils import fast_spectral_norm
from copy import copy

def rowWiseSampler(data_row, norms_ds, size, idx):    
    sparse_row = np.zeros_like(data_row)
    p1 = np.zeros_like(data_row)
    p2 = np.zeros_like(data_row)
    p3 = np.zeros_like(data_row)
    p4 = np.zeros_like(data_row)
    
    if "1" in norms_ds["mode"]:
        p1 = np.abs(data_row) / norms_ds["sum_abs_data"]
    if "2" in norms_ds["mode"]:
        p2 = (np.abs(data_row) / norms_ds["ell_one_row_norms"][idx]) * \
            (norms_ds["ell_one_row_norms_squared"][idx] / np.sum(norms_ds["ell_one_row_norms_squared"]))
    if "3" in norms_ds["mode"]:
        p3 = (np.abs(data_row) / norms_ds["ell_one_col_norms"]) * \
            (norms_ds["ell_one_col_norms_squared"] / np.sum(norms_ds["ell_one_col_norms_squared"]))
    if "4" in norms_ds["mode"]:
        p4 = (np.abs(data_row) / norms_ds["sum_abs_data"]) * \
            (norms_ds["ell_one_row_norms_squared"][idx] / np.sum(norms_ds["ell_one_row_norms_squared"]))
    
    pij_alpha = np.maximum(np.maximum(np.maximum(p1, p2), p3), p4)
    pij_alpha = np.minimum(1, size*pij_alpha)
    index_flags = np.random.rand(*pij_alpha.shape)< pij_alpha 
    
    sparse_row[index_flags] = data_row[index_flags] / (pij_alpha[index_flags])
    
    return sparse_row
    

def modifiedBKKS21(data=np.zeros((100,100)), size=10, mode="12", row_norm_preserve=True, row_norm_preserve_type="total", sparsify_op=True):
    n = len(data)
    # the divisors are off-setwith eps to make things go faster
    eps = 1e-30
    # init sparse matrix
    sparse_data = np.zeros_like(data)
    
    if row_norm_preserve:
        if row_norm_preserve_type == "total":
            total_row_sums = np.sum(data, axis=1)
    
    norms_ds = {}
    #ell_1 row norms
    norms_ds["ell_one_row_norms"] = np.linalg.norm(data, ord=1, axis=1) + eps
    norms_ds["ell_one_row_norms_squared"] =  norms_ds["ell_one_row_norms"] ** 2 + eps
    
    # ell_1 col norms
    norms_ds["ell_one_col_norms"] = np.linalg.norm(data, ord=1, axis=0) + eps
    norms_ds["ell_one_col_norms_squared"] =  norms_ds["ell_one_col_norms"] ** 2 + eps
    
    norms_ds["sum_abs_data"] = np.sum(np.abs(data)) 
    norms_ds["mode"] = mode
    
    for idx in range(len(data)):
        # sample elements per row to avoid running out of memory
        sparse_data[idx, :] = rowWiseSampler(data[idx, :], norms_ds, size, idx)
    
    if row_norm_preserve:
        if row_norm_preserve_type == "total":
            sparse_total_row_sums = np.sum(sparse_data, axis=1)
            sparse_nnz_per_rows = np.count_nonzero(sparse_data, axis=1)
            left_overs = total_row_sums - sparse_total_row_sums
            adjustments_per_row = np.zeros_like(left_overs)
            # divide only possible for rows with at least one non-zero
            adjustments_per_row[sparse_nnz_per_rows > 0] = left_overs[sparse_nnz_per_rows > 0] / sparse_nnz_per_rows[sparse_nnz_per_rows > 0] # elementwise divide
            # print(len(left_overs), len(adjustments_per_row), len(sparse_data))
            for i in range(len(sparse_data)):
                q = sparse_data[i,:]
                q[np.abs(q) > 0] = q[np.abs(q) > 0]+adjustments_per_row[i]
                sparse_data[i,:] = q
    
    if sparsify_op:
        sparse_data = scipy.sparse.csr_matrix(sparse_data)
    
    return sparse_data

def thresholdedBKKS21(data=np.zeros((100,100)), size=10, mode="12", row_norm_preserve=True, row_norm_preserve_type="total", sparsify_op=True, split_ratio=10):
    
    return

def modifiedBKKS21MemoryIntensive(data=np.zeros((100,100)), size=10, mode="12", row_norm_preserve=True, row_norm_preserve_type="total", sparsify_op=True):
    """
    code by Archan
    """
    n = len(data)
    # the divisors are off-setwith eps to make things go faster
    eps = 1e-30
    # init sparse matrix
    sparse_data = np.zeros_like(data)
    # init probability matrices
    p1 = np.zeros_like(data)
    p2 = np.zeros_like(data)
    p3 = np.zeros_like(data)
    p4 = np.zeros_like(data)
    
    if row_norm_preserve:
        if row_norm_preserve_type == "total":
            total_row_sums = np.sum(data, axis=1)
    
    #ell_1 row norms
    ell_one_row_norms = np.linalg.norm(data, ord=1, axis=1) + eps
    ell_one_row_norms_squared =  ell_one_row_norms ** 2 + eps
    
    # ell_1 col norms
    ell_one_col_norms = np.linalg.norm(data, ord=1, axis=0) + eps
    ell_one_col_norms_squared =  ell_one_col_norms ** 2 + eps
    
    # probability proportional to absolute value of each element
    if "1" in mode:
        p1 = np.abs(data) / np.sum(np.abs(data))    # sum p1 is guaranteed to be 1
    
    # probability proportional to the row norms
    if "2" in mode:
        # guaranteed to sum to 1
        p2 = (np.abs(data) / ell_one_row_norms[:, None]) * \
            ((ell_one_row_norms_squared / np.sum(ell_one_row_norms_squared))[:, None])
    
    # probability proportional to the column norms
    if "3" in mode:
        # guaranteed to sum to 1
        p3 = (np.abs(data) / ell_one_col_norms) * \
            ((ell_one_col_norms_squared / np.sum(ell_one_col_norms_squared)))
    
    if "4" in mode:
        p4 = (np.abs(data) / np.sum(np.abs(data))) * \
             ((ell_one_row_norms_squared / np.sum(ell_one_row_norms_squared))[:, None])
    
    # set up to fnd max of all the elements
    pij_alpha = np.maximum(np.maximum(np.maximum(p1, p2), p3), p4)
    # pij_alpha /= np.sum(size*pij_alpha)
    pij_alpha = np.minimum(1, size*pij_alpha)
    
    # sample values from a uniform distribution, check if they are less than the individual probabilities, if yes, then sample them
    index_flags = np.random.rand(*pij_alpha.shape) < pij_alpha 
    
    # # list of indices might be duplicate, so get weight
    # sampled_indices, weights = sample_index(pij_alpha, size)
    
    # sparse_data[sampled_indices] = data[sampled_indices]
    # sparse_data[sampled_indices] = sparse_data[sampled_indices] * weights
    # sparse_data[sampled_indices] = sparse_data[sampled_indices] / (pij_alpha[sampled_indices]) # elementwise divide
    
    sparse_data[index_flags] = data[index_flags] / (pij_alpha[index_flags]) # elementwise divide
    
    # print("norm of sparsified data:", size, np.where(sparse_data!=0))
    
    if row_norm_preserve:
        if row_norm_preserve_type == "total":
            sparse_total_row_sums = np.sum(sparse_data, axis=1)
            sparse_nnz_per_rows = np.count_nonzero(sparse_data, axis=1)
            left_overs = total_row_sums - sparse_total_row_sums
            adjustments_per_row = np.zeros_like(left_overs)
            # divide only possible for rows with at least one non-zero
            adjustments_per_row[sparse_nnz_per_rows > 0] = left_overs[sparse_nnz_per_rows > 0] / sparse_nnz_per_rows[sparse_nnz_per_rows > 0] # elementwise divide
            # print(len(left_overs), len(adjustments_per_row), len(sparse_data))
            for i in range(len(sparse_data)):
                q = sparse_data[i,:]
                q[np.abs(q) > 0] = q[np.abs(q) > 0]+adjustments_per_row[i]
                sparse_data[i,:] = q
    
    if sparsify_op:
        sparse_data = scipy.sparse.csr_matrix(sparse_data)
    return sparse_data


def row_operation(copy_row, threshold):
    """
    code from RMR
    """
    argzero = np.argwhere((np.abs(copy_row) <= threshold) * (copy_row != 0))
    argzero = argzero.reshape(len(argzero),)
    argzero_copy = copy_row[argzero]
    copy_row[argzero] = 0
    sum = np.sum(argzero_copy)
    if sum != 0:
        k = math.ceil(sum / threshold)
        indices = np.random.choice(argzero, k, p=argzero_copy/sum, replace=True)
        np.add.at(copy_row, indices, sum / k)

def RMR(matrix=np.zeros((100,100)), threshold=0.05, sparsify_op=True):
    """
    code from RMR
    """
    threshold = threshold * np.max(matrix)
    print("Cut-off threshold:", threshold)
    copy_matrix = matrix.copy()
    np.apply_along_axis(row_operation, 1, copy_matrix, threshold)
    # print("**************checking***************", copy_matrix.shape)
    # print("**************checking***************", fast_spectral_norm(copy_matrix))
    if sparsify_op:
        copy_matrix = scipy.sparse.csr_matrix(copy_matrix)
    return copy_matrix


def AHK06(matrix=np.zeros((100,100)), threshold=0.05, sparsify_op=True):
    """
    code from RMR
    """
    copy_matrix = matrix.copy()
    n, d = matrix.shape
    probs = np.random.random((n, d))
    copy_matrix[np.abs(matrix) < threshold] = 0
    copy_matrix[probs < (np.abs(matrix) / threshold) * (np.abs(matrix) < threshold)] = threshold
    # copy_matrix = scipy.sparse.csr_matrix(copy_matrix)
    if sparsify_op:
        copy_matrix = scipy.sparse.csr_matrix(copy_matrix)
    return copy_matrix

def AHK06_true(matrix=np.zeros((100,100)), threshold=0.05, sparsify_op=True):
    """
    code by Archan
    """
    copy_matrix = matrix.copy()
    n, d = matrix.shape
    copy_matrix[np.abs(matrix) <= threshold] = 0
    mask = np.ones_like(matrix)
    mask[matrix > threshold] = 0
    sign_matrix = np.ones_like(matrix)
    sign_matrix[matrix < 0] = -1
    probs = (np.abs(matrix) / threshold) * mask # probabilities for independent sampling
    index_flags = np.random.rand(*probs.shape) < probs
    copy_matrix[index_flags] = sign_matrix[index_flags]*threshold
    # copy_matrix = scipy.sparse.csr_matrix(copy_matrix)
    if sparsify_op:
        copy_matrix = scipy.sparse.csr_matrix(copy_matrix)
    return copy_matrix


def compute_row_distribution(matrix, s, delta, row_norms):
    """
    code from RMR
    """
    m, n = matrix.shape
    z = row_norms / np.sum(row_norms)
    alpha, beta = math.sqrt(np.log((m + n) / delta) / s), np.log((m + n) / delta) / (3 * s)
    zeta = 1
    rou = (alpha * z / (2 * zeta) + ((alpha * z / (2 * zeta)) ** 2 + beta * z / zeta) ** (1 / 2)) ** 2
    sum = np.sum(rou)
    while np.abs(sum - 1) > 1e-5:
        zeta *= sum
        rou = (alpha * z / (2 * zeta) + ((alpha * z / (2 * zeta)) ** 2 + beta * z / zeta) ** (1 / 2)) ** 2
        sum = np.sum(rou)
    return rou

def AKL13(data=np.zeros((100,100)), size=100, sparsify_op=True):
    """
    code from RMR
    """
    data = data.T
    size = int(size)
    n, d = data.shape
    row_norms = np.linalg.norm(data, axis=1, ord=1)
    rou = compute_row_distribution(data, size, 0.1, row_norms)
    nonzero_indices = data.nonzero()
    matrix = data[nonzero_indices]
    row_norms[row_norms == 0] = 1
    probs_matrix = rou.reshape((n, 1)) * data / row_norms.reshape((n, 1))
    probs = probs_matrix[nonzero_indices]
    probs /= np.sum(probs)
    indices = np.arange(len(matrix))
    selected = np.random.choice(indices, size, p=probs, replace=True)
    result = np.zeros((n, d))
    np.add.at(result, (nonzero_indices[0][selected], nonzero_indices[1][selected]), matrix[selected] / (probs[selected] * size))
    result = result.T
    data = data.T
    # result = scipy.sparse.csr_matrix(result)
    if sparsify_op:
        result = scipy.sparse.csr_matrix(result)
    return result

def DZ11(matrix=np.zeros((100,100)), threshold=0.05, sparsify_op=True):
    """
    code from RMR
    """
    copy_matrix = matrix.copy()
    n, d = matrix.shape
    norm_fro = np.linalg.norm(matrix, ord="fro")
    copy_matrix[np.abs(matrix) <= threshold / (n + d)] = 0
    s = int(14 * (n + d) * np.log(np.sqrt(2) / 2 * (n + d)) * (norm_fro / threshold) ** 2)
    nonzero_indices = copy_matrix.nonzero()
    data = copy_matrix[nonzero_indices]
    probs_matrix = copy_matrix * copy_matrix
    probs = probs_matrix[nonzero_indices]
    probs /= np.sum(probs)
    indices = np.arange(len(data))
    selected = np.random.choice(indices, s, p=probs, replace=True)
    result = np.zeros((n, d))
    np.add.at(result, (nonzero_indices[0][selected], nonzero_indices[1][selected]), data[selected] / (probs[selected] * s))
    # result = scipy.sparse.csr_matrix(result)
    if sparsify_op:
        result = scipy.sparse.csr_matrix(result)
    return result

def heavyRMR(matrix, threshold, max_samples=4000000, sparsify_op=True):
    copy_matrix = np.zeros_like(matrix)
    # remove all elements below threshold
    copy_matrix[np.abs(matrix) > threshold] = matrix[np.abs(matrix) > threshold]
    residual_matrix = matrix - copy_matrix
    sample_budget = max_samples - np.count_nonzero(copy_matrix)
    # print(sample_budget)
    # print(np.sum(residual_matrix))
    
    # picks and reweights small entries
    if sample_budget > 0:
        new_residual = modifiedBKKS21(residual_matrix, size=sample_budget, mode="2")
        copy_matrix = scipy.sparse.csr_matrix(copy_matrix)
        copy_matrix += new_residual
        
    if sparsify_op:
        copy_matrix = scipy.sparse.csr_matrix(copy_matrix)

    return copy_matrix

def noSparse(matrix, threshold=0.0):
    matrix = scipy.sparse.csr_matrix(matrix)
    return matrix