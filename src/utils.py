import numpy as np

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
    
def sample_index(probs, s):
    indices = np.random.choice(np.arange(probs.size), p=probs.ravel(), replace=True, size=s)
    return np.unravel_index(indices, probs.shape)