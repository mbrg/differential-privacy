import numpy as np


def normalize(v, ord=1):
    norm = np.linalg.norm(v, ord=ord)
    if norm == 0:
       return v
    return v / norm


def norm(v, ord=1):
    try:
        n = np.linalg.norm(v, ord)
    except ValueError:
        n = v[0] if isinstance(v, (np.ndarray, tuple, list)) else v
    return n
