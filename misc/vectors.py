import numpy as np


def normalize(v, norm_ord=1):
    norm = np.linalg.norm(v, ord=1)
    if norm == 0:
       return v
    return v / norm
