"""
A simplified QUBO solving algorithm.
"""
import numpy as np
from neva.tools.QUBO_tools import QUBO_Value


def step(x: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    One step of the algorithm in O(m^2) where Q is
    in M(m, m)
    """
    return(np.minimum(np.ones(x.shape), np.maximum(np.zeros(x.shape),np.matmul(Q, x))))

"""
The main loop, that runs in O(m^2*n) on a classical
von Neumann architecture
"""
def deterministQUBO(Q):
    """
    A simplified QUBO solving algorithm.
    """
    N = Q.shape[0]
    x = np.random.random((N,))
    for i in range(N):
        x = step(x, Q)

    return QUBO_Value(x=x, Q=Q)
