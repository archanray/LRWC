import numpy as np

def count_elements_for_threshold(row, T, d, sum_mode="ell_one"):
    """
    check per row total count of elements not exceeding the threshold
    """
    sorted_row = sorted(row)
    current_sum = 0
    count = 0
    for j in range(d,-1,-1):
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

m = 12
np.random.seed(0)
A = np.abs(np.random.rand(5,5))
print("input\n", A)
print("threshold:", find_threshold(A, m))