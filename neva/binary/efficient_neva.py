"""
Implementation of a Cellular Genetic 
Algorithm for optimization with matrices
"""

import numpy as np
import matplotlib.pyplot as plt
from neva.tools.QUBO_tools import QUBO_Value, sparse_to_array
from neva.tools.SAT_Tools import cnf_to_sat, evaluate
from typing import Dict, List, Tuple
import time

plt.style.use("fivethirtyeight")
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



def run_spk(k, N, s_c, s_d, tau, t, r, datas, tau_max, Combine, Mutate, f):
    """
    One time step of computation for the NEVA algorithm
    """
    for n in range(len(N)):
        if t[n] <= 0:
            for m in N[n]:
                s_c[m] += 1
                s_d[m] += datas[n]
            tau[n] += np.random.randint(2)
            t[n] = tau[n]
            if tau[n] > tau_max:
                datas[n] = Mutate(datas[n])
                tau[n] = 0
        else:
            t[n] -= 1
    for n in range(len(N)):
        if r[n] <= 0 and s_c[n] == 1:
            temp = Combine(s_d[n], datas[n])
            if f(temp) > f(datas[n]):
                datas[n] = temp
                r[n] = np.random.randint(0, k + 1)
        elif r[n] > 0:
            r[n] -= 1
        s_c[n] = 0
        s_d[n] = np.zeros(datas[0].shape)


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


def CGA_simple(V:List[int], E:List[Tuple[int, int]], k:int,  f, num_steps:int, D: int, max_period:int=5, Combine=combine1, Mutate=mutate1,  probe:bool=False):
    """
    Computes the NEVA algorithm ending datas in an array through regular matrices
    ------------------
    V : Set of all vertices, must be {0,...,N-1}
    E : Set of all ridges in the interaction graph
    k : Waiting time after successfull combination
    Combine : Array[bool] * Array[bool] -> Array[bool] Function used for combining solutions
    Mutate : Array[bool] -> Array[bool] Function used for mutating solutions
    f : Array[bool] -> float Function to optimize
    max_period : Period before the neuron starts mutating
    D : Dimensionnality of the problem
    probe : If set to True, CGA_simple now returns all computed datas at all time
    """
    N = graph_to_N(E, V)
    s_c = [0 for _ in V]
    s_d = [np.zeros(D) for _ in V]
    tau = [0 for _ in V]
    t = [0 for _ in V]
    r = [0 for _ in V]
    datas = [np.random.random(size=(D,)) <= 0.5 for _ in V]
    if probe:
        d = [[] for _ in V]
    for _ in range(num_steps):
        run_spk(
            k=k,
            N=N,
            s_c=s_c,
            s_d=s_d,
            tau=tau,
            t=t,
            r=r,
            datas=datas,
            tau_max=max_period,
            Combine=Combine,
            Mutate=Mutate,
            f=f,
        )
        if probe:
            for i in V:
                d[i].append(f(datas[i]))
    if probe:
        return d
    else:
        return datas



