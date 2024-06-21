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


def simple_annealing(problem, D, n: int, temperature, s=None, historic=False):
    """
    Returns the best QUBO value in O(n³) time
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


def torus(n: int) -> Tuple[List[int], List[Tuple[int, int]]]:
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


"""
Parameters
"""

pb= "gka4e" # "uf200-06"
Q = sparse_to_array(f"gka_sparse_all/{pb}.sparse")  # Import benchmark instances as a numpy array
# sat = cnf_to_sat(f"SAT/uf200-860/{pb}.cnf")
N = 64
V, E =  torus(N) # ring_one_way(N)
probe = 0
s = 5  # Number of step to wait before combining again
D = Q.shape[0] # 200
figname= f"No_mutation_{pb}_N={N}_D={D}"
memetic = False # None to enable figname over automatic
problem = lambda x: QUBO_Value(Q, x) # evaluate(sat, x)  # Binary problem to solve
combination = lambda x, y: combine1(x, y)  # Method for combining solutions
mutate = lambda x: mutate1(x, k=(5/100)*D) # if not memetic else (mutate1(x, k=np.log(D)) if np.random.random() >= 0.1 else mutate3(x, Q, 1))  # simple_annealing(Q, 200, temperature=lambda x:x**2, s=x)                          # Method for mutating
num_steps = 200  # Number of steps to run the swarm for
k = 4  # Max range for waiting time
max_period = 5  # Period before the particle starts mutating
f0 = (lambda x: x)  # mutate3(x, Q, 50)                     # Initialisation of positionning
p_err = 0  # Probability of combining even if the rsult will be less

"""
End of Parameters
"""


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


if __name__ == "__main__":
    begin = time.time()
    d = CGA_simple(
        V=V,
        E=E,
        k=k,
        D=D,
        max_period=max_period,
        Combine=combination,
        Mutate=mutate,
        f=problem,
        num_steps=num_steps,
        probe=True,
    )
    end = time.time()
    print(end-begin)
    plt.figure(figsize=(10, 8))
    plt.plot([max([d[i][j] for i in V]) for j in range(num_steps)])
    plt.plot([np.average([d[i][j] for i in V]) for j in range(num_steps)])
    plt.plot(simple_annealing(problem, D, num_steps, lambda x: x, historic=True))
    plt.xlabel("Time steps")
    plt.ylabel("Max(X)|E(F(X))")
    plt.tight_layout()
    if memetic is None:
        plt.savefig(figname)
    elif memetic:
        plt.savefig(f"graphs/NEVA_Memetic_Max(X)|E(F(X))_{pb}_N={N}_D={D}")
    else:
        plt.savefig(f"graphs/NEVA_Max(X)|E(F(X))_{pb}_N={N}_D={D}")
    # for i in V:
    #     plt.plot(d[i])
    # print(max([problem(d) for d in CGA_simple(V, E, k, max_period=max_period, Combine=combination, Mutate=mutate, f=problem, num_steps=num_steps)]))
    # plt.show()
    print(max([v[num_steps - 1] for v in d]))
