import numpy as np
from math import exp

def QUBO_Value(Q, x):
    """
    Given a QUBO matrix Q and a unitary vector x,
    returns the value of subset x for problem Q
    """
    return np.matmul(x, np.matmul(Q,np.transpose(x)))

def bound(f: float, a: float, b: float) -> float:
    """
    Set f to a if f < a and to b if f > b
    """
    return min(max(f, a), b)

def sparse_to_array(filename: str) -> np.ndarray:
    """
    Reads out a .sparse QUBO file and converts
    it to a numpy array
    """
    mat = open(filename, "r")
    n, m = [int(i) for i in mat.readline().split(' ')]
    Q = np.zeros((n,n))
    for _ in range(m):
        i, j, q = [int(i) for i in mat.readline().split(' ')]
        Q[i - 1, j - 1] = -q
        Q[j - 1, i - 1] = -q
    return Q

def degree(a: int, p: np.ndarray) -> int:
    d = 0
    for i in p[a]:
        d+=i
    return d

def partitionning_to_QUBO(p: np.ndarray) -> np.ndarray:
    """
    Given an array that represents a graph, returns the QUBO
    problem corresponding to it's Newmann modularity
    """
    N = p.shape[0]
    q = np.array([degree(i, p) for i in range(N)])
    m = q.sum()
    return p - np.matmul(np.transpose(q), q)/m

def QUBO_random_solver(Q: np.ndarray, n: int) -> float:
    """
    Randomly picks n vectors and returns the best
    QUBO value found
    """
    maxi = -float("inf")
    s = (Q.shape[0], )
    for _ in range(n):
        maxi = max(QUBO_Value(Q, np.random.random(size=s)<=0.5), maxi)
    return maxi

def QUBO_annealing(Q: np.ndarray, n: int, temperature=lambda x:x):
    """
    Returns an approached maximum QUBO value
    using 
    """
    N = Q.shape[0]
    s = np.random.random((N, )) <= 0.5
    e = QUBO_Value(Q, s)
    for k in range(n):
        T = temperature(1-k/n)
        k = np.random.randint(0, N)
        snew = s.copy()
        snew[k] = False if s[k] else True
        enew = QUBO_Value(Q, snew)
        if np.random.random() <= (1 if enew > e else exp(-(e - enew)/T)):
            s = snew
            e = enew
    return e


def simulated_annealing(problem, D:int, n: int, temperature=lambda x:x, s:np.ndarray=None, historic:bool=False):
    """
    Simulated annealing implementation for maximisation
    -----------
    problem : Function to maximize
    D : Dimension of the problem
    n : Number of steps to run the annealing for
    temperature : [0,1] -> [0,1] Temperature used for annealing
    s : Starting point
    historic : If set to to true, returns a list of all best found values at
    each time step
    """
    tout = []
    if s is None:
        s = np.random.random((D,)) <= 0.5
    e = problem(s)
    smax = s
    emax = e
    for k in range(n):
        T = temperature(1 - k / n)
        k = np.random.randint(0, D)
        snew = s.copy()
        snew[k] = False if s[k] else True
        enew = problem(snew)
        if np.random.random() <= (1 if enew > e else np.exp(-(e - enew) / T)):
            s = snew
            e = enew
        if enew > emax:
            smax = snew
            emax = enew
        tout.append(emax)
    if historic:
        return tout
    else:
        return emax
