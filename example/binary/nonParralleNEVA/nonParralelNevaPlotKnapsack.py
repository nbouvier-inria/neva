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
from neva.tools.knapsack_tools import knap_to_problem, knap_to_problem_penalisation
from neva.tools.CGA_tools import torus, mutate1, mutate2, mutate3
plt.style.use("fivethirtyeight")
from typing import List, Tuple

"""
Parameters
"""
pb= "knapPI_1_1000_1000_1" 
# Import benchmark instances as a numpy array
filename = f"./benchmarks/instances_01_KP/large_scale/{pb}"  
# Number of individuals
N = 256
# Interaction graph
V, E =  torus(N) # ring_one_way(N)
# Dimension of the problem
D = 1000


# Binary problem to solve
problem = knap_to_problem_penalisation(filename=filename) # lambda x : evaluate(sat, x) for sat example
# Method for combining solutions
combination = lambda x, y: combine1(x, y)  
# Method for mutating
mutate = lambda x: mutate1(x, k=(1/1000)*D) 

num_steps = 2000  # Number of steps to run the swarm for
k = 4  # Max range for waiting time
max_period = 5  # Period before the particle starts mutating
f0 = (lambda x: x)  # mutate3(x, Q, 50)                     # Initialisation of positionning
p_err = 0  # Probability of combining even if the rsult will be less
g = knap_to_problem(filename=filename) # Function to calculate results on
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
        f0=lambda x:np.zeros(shape=(len(x),)),
        g=lambda x:x
    )
    end = time.time()
    print("Computation time : ", round(end-begin, 3))
    plt.figure(figsize=(10, 8))
    histo_f = [max([(g(d[i][j])) for i in V]) for j in range(num_steps)]
    histo_f_etoile = [max([(problem(d[i][j])) for i in V]) for j in range(num_steps)]
    plt.plot(histo_f_etoile)
    plt.plot([np.average([problem(d[i][j]) for i in V]) for j in range(num_steps)])
    plt.plot(simulated_annealing(problem, D, num_steps, lambda x: x, historic=True, s=np.zeros(shape=(D),)))
    plt.plot([i if i > 1 else -float("inf") for i in histo_f])
    plt.xlabel("Time steps")
    plt.ylabel("Max(f*(X))|E(f*(X))|Max(f(X))")
    plt.tight_layout()
    plt.savefig(f"../graphs/NEVA_Knapsack_{pb}")

    # plt.savefig(f"../graphs/NEVA_Max(X)|E(F(X))_{pb}_N={N}_D={D}")
    # for i in V:
    #     plt.plot(d[i])
    # print(max([problem(d) for d in CGA_simple(V, E, k, max_period=max_period, Combine=combination, Mutate=mutate, f=problem, num_steps=num_steps)]))
    # plt.show()
    print("Maximum found value is :", max(histo_f))