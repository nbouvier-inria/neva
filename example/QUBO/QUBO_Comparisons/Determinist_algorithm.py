import numpy as np
from neva.QUBO_tools import QUBO_Value, sparse_to_array, bound, QUBO_random_solver, QUBO_annealing

Q = sparse_to_array('gka_sparse_all/gka5e.sparse')

"""
Q = np.array([[0, 5, 0, -3, 7],
              [5, 0, -1, 0, 0],
              [0, -1, 0, -2, 0],
              [-3, 0, -2, 0, 0],
              [7, 0, 0, 0, 0]], dtype=int)
"""

def step(x: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    One step of the algorithm in O(m²) where Q is
    in M(m, m)
    """
    return(np.minimum(np.ones(x.shape), np.maximum(np.zeros(x.shape),np.matmul(Q, x))))

"""
The main loop, that runs in O(m²*n) on a classical
von Neumann architecture
"""
def deterministic_solve(Q, n):
    """
    Time: O(d²*n)
    Process: O(1) 
    (where e = num_explos, s = num_snn, n = num_steps, d = dimension)
    """
    N = n
    x = np.random.random((N,))
    for i in range(N):
        x = step(x, Q)
    return QUBO_Value(x=x, Q=Q)
