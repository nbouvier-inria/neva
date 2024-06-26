"""
Implementation of a Cellular Genetic 
Algorithm for optimization with matrices
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from neva.tools.CGA_tools import mutate1, combine1, graph_to_N

plt.style.use("fivethirtyeight")
from typing import List, Tuple




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

def nonParrallelNeva(V:List[int], E:List[Tuple[int, int]], f, num_steps:int, D: int, k:int=4, max_period:int=5, Combine=combine1, Mutate=mutate1,  probe:bool=False, f0=lambda x:x, g=None):
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
    f0 : Array[bool] -> Array[bool] Function applied to the first instance
    g : Array[bool] -> float Function to use for probing results 
    """
    N = graph_to_N(E, V)
    s_c = [0 for _ in V]
    s_d = [np.zeros(D) for _ in V]
    tau = [0 for _ in V]
    t = [0 for _ in V]
    r = [0 for _ in V]
    datas = [f0(np.random.random(size=(D,)) <= 0.5) for _ in V]
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
                d[i].append(f(datas[i]) if g is None else g(datas[i]))
    if probe:
        return d
    else:
        return datas



