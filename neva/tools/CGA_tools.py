import numpy as np
from typing import List, Tuple

def mutate1(x: np.ndarray, k: int = 1):
    """
    Random uniform k bitflips on array x
    """
    return np.logical_xor(x, (np.random.random(x.shape) <= k / x.shape[0]))


def mutate2(x: np.ndarray, Q: np.ndarray):
    """
    One step of the naive QUBO heuristic
    Q : QUBO matrix
    x : Current solution
    """
    return np.minimum(np.ones(x.shape), np.maximum(np.zeros(x.shape), np.matmul(Q, x)))


def mutate3(x: np.ndarray, Q: np.ndarray, n: int):
    """
    n steps of the naive QUBO heuristic
    Q : QUBO matrix
    x : Current solution
    """
    if n > 0:
        return mutate3(mutate2(x, Q), Q, n - 1)
    else:
        return x
    
def dist(x: np.ndarray, y: np.ndarray):
    D = x.shape[0]
    diff = 0
    for i in range(D):
        if x[i] != y[i]:
            diff += 1
    return diff / D * 100


def ring_one_way(n: int) -> Tuple[List[int], List[Tuple[int, int]]]:
    V = [i for i in range(n)]
    E = [(0, n - 1)]
    for i in range(n - 1):
        E.append((i, i + 1))
    return (V, E)


def grid(n: int) -> Tuple[List[int], List[Tuple[int, int]]]:
    n = int(np.sqrt(n))
    V = [i for i in range(n**2)]
    E = []
    for i in range(n - 1):
        E.append((i + n * (n - 1), i + 1 + n * (n - 1)))
        E.append((n - 1 + n * (i), n - 1 + n * (i + 1)))
        for j in range(n - 1):
            E.append((i + n * j, i + n * j + 1))
            E.append((i + n * j, i + n * (j + 1)))
    return V, E


def torus(n: int) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Square orus-shaped graph (V, E) of
    less than n vertices
    """
    n = int(np.sqrt(n))
    V = [i for i in range(n**2)]
    E = []
    for i in range(n - 1):
        E.append((i + n * (n - 1), i + 1 + n * (n - 1)))
        E.append((i + n * (n - 1), i))
        E.append((n - 1 + n * (i), n - 1 + n * (i + 1)))
        E.append((n - 1 + n * (i), 1 + n * (i)))
        for j in range(n - 1):
            E.append((i + n * j, i + n * j + 1))
            E.append((i + n * j, i + n * (j + 1)))
    return V, E






def graph_to_N(E, V):
    """
    Returns the non-directed neighbourhood based
    on graph G = (V, E)
    """
    N = [[] for _ in V]
    for i, j in E:
        N[i].append(j)
        N[j].append(i)
    return N


def combine1(x: np.ndarray, y: np.ndarray):
    """
    Uniform random combination of x and y
    """
    m = np.random.random(x.shape) < 0.5
    return x * m + y * (1 - m)


def combine2(x: np.ndarray, y: np.ndarray):
    """
    One point combination of x and y
    """
    m = int(np.random.random() * x.shape[0])
    retour = np.zeros(x.shape)
    for i in range(m):
        retour[i] = x[i]
    for i in range(m, x.shape[0]):
        retour[i] = y[i]
    return retour