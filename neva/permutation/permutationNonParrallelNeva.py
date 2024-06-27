"""
Implementation of a Cellular Genetic 
Algorithm for optimization with matrices
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from neva.tools.CGA_tools import graph_to_N

plt.style.use("fivethirtyeight")
from typing import List, Tuple




def run_spk(value, pre, send, C, N, tau, tau_max, t, Mutate, data:List[np.ndarray], f,time_step, T=None ):
    """
    One time step of computation for the NEVA algorithm
    """
    D = data[0].shape[0]
    for n in np.random.permutation(len(N)):
        if t[n] <= 0:
            if send[n] >= D:
                send[n] = 0
                tau[n] += np.random.randint(2)
                t[n] = tau[n]
                if tau[n] > tau_max:
                    data[n] = Mutate(data[n])
            else:
                c = np.where(data[n] == send[n])[0][0]
                for m in N[n]:
                    if pre[m][c] is None:
                        pre[m][c] = C[m]
                        C[m] += 1
                send[n] += 1
        else:
            t[n] -= 1
    for n in range(len(N)):
        if C[n] >= D:
            v = f(pre[n])
            if v >= value[n] or (np.exp(-(value[n]-v)/(T(time_step)*abs(value[n]))) > np.random.random() if T is not None else False):
                data[n] = pre[n]
                value[n] = v
                tau[n] = 0
            pre[n] = np.array([None for _ in range(D)])
            C[n] = 0


def nonParrallelNevaPermutation(V:List[int], E:List[Tuple[int, int]], D, f, num_steps:int, tau_max, Mutate, T=None,  probe:bool=False, f0=lambda x:x):
    """
    Computes the NEVA algorithm ending datas in an array through regular matrices
    ------------------
    V : Set of all vertices, must be {0,...,N-1}
    E : Set of all ridges in the interaction graph
    f : Array[bool] -> float Function to optimize
    T : int -> [0,1] Temperature function given time
    num_steps : int Number of stes to run the algorithm for
    D : Dimensionnality of the problem
    probe : If set to True, CGA_simple now returns all computed datas at all time
    f0 : Array[bool] -> Array[bool] Function applied to the first instance
    """
    N = graph_to_N(E, V)
    datas = [f0(np.random.permutation(D)) for _ in V]
    value = [f(d) for d in datas]
    t = [0 for _ in V]
    tau = [0 for _ in V]
    pre = [np.array([None for _ in range(D)]) for _ in V]
    send = [0 for _ in V]
    C = [0 for _ in V]
    if probe:
        d = [[] for _ in V]
    for step in range(num_steps):
        run_spk(
            T=T,
            N=N,
            data=datas,
            tau=tau,
            time_step=step,
            value=value,
            f=f,
            pre=pre,
            send=send,
            t=t,
            C=C,
            tau_max=tau_max,
            Mutate=Mutate
        )
        if probe:
            for i in V:
                d[i].append(f(datas[i]))
    if probe:
        return d
    else:
        return datas