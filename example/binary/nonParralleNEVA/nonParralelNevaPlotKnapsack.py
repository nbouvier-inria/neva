"""
An example scipt for plotting results
on QUBO instances
"""
import numpy as np
import matplotlib.pyplot as plt
from neva.binary.nonParrallelNeva import *
from typing import Dict, List, Tuple
import time

from neva.tools.QUBO_tools import simulated_annealing
from neva.tools.knapsack_tools import knap_to_problem
from neva.tools.CGA_tools import torus, mutate1, mutate2, mutate3
plt.style.use("fivethirtyeight")
from typing import List, Tuple

"""
Parameters
"""
pb= "large_scale/knapPI_1_1000_1000_1" 
# Import benchmark instances as a numpy array
filename = f"../instances_01_KP/{pb}"  
# Number of individuals
N = 256
# Interaction graph
V, E =  torus(N) # ring_one_way(N)
# Dimension of the problem
D = 10000


# Binary problem to solve
problem = knap_to_problem(filename=filename) # lambda x : evaluate(sat, x) for sat example
# Method for combining solutions
combination = lambda x, y: combine1(x, y)  
# Method for mutating
mutate = lambda x: mutate1(x, k=(1/1000)*D) 

num_steps = 500  # Number of steps to run the swarm for
k = 4  # Max range for waiting time
max_period = 5  # Period before the particle starts mutating
f0 = (lambda x: x)  # mutate3(x, Q, 50)                     # Initialisation of positionning
p_err = 0  # Probability of combining even if the rsult will be less
"""
End of Parameters
"""

if __name__ == "__main__":
    begin = time.time()
    print("Running...")
    d = nonParrallelNeva(
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
        f0=lambda x:np.zeros(shape=(len(x),))
    )
    end = time.time()
    print("Computation time : ", round(end-begin, 3))
    plt.figure(figsize=(10, 8))
    plt.plot([max([d[i][j] for i in V]) for j in range(num_steps)])
    plt.plot([np.average([d[i][j] for i in V]) for j in range(num_steps)])
    plt.plot(simulated_annealing(problem, D, num_steps, lambda x: x, historic=True, s=np.zeros(shape=(D),)))
    plt.xlabel("Time steps")
    plt.ylabel("Max(X)|E(F(X))")
    plt.tight_layout()
    plt.show()

    # plt.savefig(f"../graphs/NEVA_Max(X)|E(F(X))_{pb}_N={N}_D={D}")
    for i in V:
        plt.plot(d[i])
    # print(max([problem(d) for d in CGA_simple(V, E, k, max_period=max_period, Combine=combination, Mutate=mutate, f=problem, num_steps=num_steps)]))
    plt.show()
    print(max([v[num_steps - 1] for v in d]))