"""
An example scipt for plotting results
on QUBO instances
"""
import numpy as np
import matplotlib.pyplot as plt
from neva.binary.nonParrallelNeva import *
from typing import Dict, List, Tuple
import time

from neva.tools.QUBO_tools import simulated_annealing, sparse_to_array,  QUBO_Value
from neva.tools.CGA_tools import torus, mutate1, mutate2, mutate3, ea
plt.style.use("fivethirtyeight")
from typing import List, Tuple

"""
Parameters
"""
pb= "gka4e" # "uf200-06"
# Import benchmark instances as a numpy array
filename = f"../gka_sparse_all/{pb}.sparse"  
# Set the QUBO problem for Q
Q = sparse_to_array(filename=filename)  # sat = cnf_to_sat(f"SAT/uf200-860/{pb}.cnf")
# Number of individuals 
N = 128
# Interaction graph
V, E =  torus(N) # ring_one_way(N)
# Dimension of the problem
D = Q.shape[0] # 200 for sat example

# True for memetic neva, False for regular neva and None for personnalisation
memetic = False
# Optionnal figure name if memetic is None
figname= f"../No_mutation_{pb}_N={N}_D={D}"
# Binary problem to solve
problem = lambda x: QUBO_Value(Q, x) # lambda x : evaluate(sat, x) for sat example
# Method for combining solutions
combination = lambda x, y: combine1(x, y)  
# Proportion of bits to flip
gamma = 5/100
# Method for mutating
mutate = lambda x: mutate1(x, k=gamma*D) if not memetic else (mutate1(x, k=np.log(D)) if np.random.random() >= 0.1 else mutate3(x, Q, 1)) 

num_steps = 200  # Number of steps to run the swarm for
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
    )
    end = time.time()
    print("Computation time : ", round(end-begin, 3))
    plt.figure(figsize=(10, 8))
    plt.plot([max([d[i][j] for i in V]) for j in range(num_steps)])
    plt.plot([np.average([d[i][j] for i in V]) for j in range(num_steps)])
    plt.plot(simulated_annealing(problem, D, num_steps, lambda x: x, historic=True))

    mu = int(0.1*N)
    lambda_ = N-mu
    h = ea(problem=problem, D=D, N=N, num_steps=num_steps, cxpb=0.3, mutpb=0.7, lambda_=lambda_, mu=mu, f0=f0, gamma=gamma)
    plt.plot(h)
    plt.xlabel("Time steps")
    plt.ylabel("Max(X)|E(F(X))")
    plt.tight_layout()
    plt.show()
    if memetic is None:
        plt.savefig(figname)
    elif memetic:
        plt.savefig(f"../graphs/NEVA_Memetic_Max(X)|E(F(X))_{pb}_N={N}_D={D}")
    else:
        plt.savefig(f"../graphs/NEVA_Max(X)|E(F(X))_{pb}_N={N}_D={D}")
    # for i in V:
    #     plt.plot(d[i])
    # print(max([problem(d) for d in CGA_simple(V, E, k, max_period=max_period, Combine=combination, Mutate=mutate, f=problem, num_steps=num_steps)]))
    # plt.show()
    print("Maximum found value is :", max([v[num_steps - 1] for v in d]))