import numpy as np

def tsp_compute(x:np.ndarray, Q:np.ndarray):
    tot = 0
    i = x[x.shape[0] - 1]
    for j in x:
        tot += Q[i, j]
        i = j
    return tot
