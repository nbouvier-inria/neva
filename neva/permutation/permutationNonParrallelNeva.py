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
from tqdm import tqdm




def run_spk(value, pre, send, C, N, tau, tau_max, t, Mutate, Combine, data:List[np.ndarray], f,time_step, T=None, Meme=None, p_meme=1):
    """
    One time step of computation for the NEVA algorithm
    """
    D = data[0].shape[0]
    for n in range(len(N)): # np.random.permutation(len(N)):
        if t[n] <= 0:
            if send[n] >= D:
                send[n] = 0
                if tau[n] < 2*tau_max*D:
                    tau[n] += np.random.randint(2*D)
                else:
                    tau[n] = max(np.random.randint(2*D), tau[n]-np.random.randint(2*D))
                t[n] = tau[n]
                # if tau[n] > tau_max*D:
                #     temp = Mutate(data[n])
                #     if f(temp) >= value[n]:
                #         data[n] = temp.copy()
                #         tau[n] = np.random.randint(D)
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
            # if n == 0:
            #     print(pre[n])
            if np.random.random() < p_meme:
                pre[n] = Meme(pre[n][:np.random.randint(D)])
            pre[n] = Combine(pre[n], data[n])
            if np.random.random() < 0.5:
                pre[n] = Mutate(pre[n])
            v = f(pre[n])
            if v >= value[n]: # or (np.exp(-(value[n]-v)/(T(time_step)*abs(value[n]))) > np.random.random() if T is not None else False):
                data[n] = pre[n].copy()
                value[n] = v
                tau[n] = np.random.randint(D)
            pre[n] = np.array([None for _ in range(D)])
            C[n] = 0


def nonParrallelNevaPermutation(V:List[int], E:List[Tuple[int, int]], D, f, num_steps:int, tau_max, Mutate, Combine=(lambda x, y:x), T=None,  probe:bool=False, f0=lambda x:x, Meme=None, p_meme=1):
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
    maxi = -float("inf")
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
    for step in tqdm(range(num_steps)):
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
            Mutate=Mutate,
            Combine=Combine,
            Meme=Meme,
            p_meme=p_meme
        )
        if probe:
            for i in V:
                temp = f(datas[i])
                d[i].append(temp)
                if temp > maxi:
                    maxi = temp
                    # print(f"New maximum of value {maxi} found !")
    if probe:
        return d
    else:
        return datas