import numpy as np
import matplotlib.pyplot as plt
from neva.binary.efficient_neva import *
from typing import Dict, List, Tuple
import time
from neva.tools.CGA_tools import mutate1, mutate3, torus
from neva.tools.SAT_Tools import cnf_to_sat, evaluate

plt.style.use("fivethirtyeight")
from typing import List, Tuple

"""
Parameters
"""
pb= "uf200-06"

# Import benchmark instances as a numpy array
sat = cnf_to_sat(f"SAT/uf200-860/{pb}.cnf")
# Number of individuals
N = 64
# Interaction graph
V, E =  torus(N) # ring_one_way(N)
# Number of step to wait before combining again
s = 5  
# Dimension of the problem
D = 200

# True for memetic neva, False for regular neva and None for personnalisation
memetic = False
# Optionnal figure name if memetic is None
figname= f"No_mutation_{pb}_N={N}_D={D}"
# Binary problem to solve
problem = lambda x : evaluate(sat, x)
# Method for combining solutions
combination = lambda x, y: combine1(x, y)  
# Method for mutating
mutate = lambda x: mutate1(x, k=(5/100)*D) if not memetic else (mutate1(x, k=np.log(D)) if np.random.random() >= 0.1 else mutate3(x, Q, 1)) 

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
    d = efficient_neva(
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
    print("Computation time : ", round(end-begin, 2) , "s")
    plt.figure(figsize=(10, 8))
    plt.plot([max([d[i][j] for i in V]) for j in range(num_steps)])
    plt.plot([np.average([d[i][j] for i in V]) for j in range(num_steps)])
    plt.plot(simulated_annealing(problem, D, num_steps, lambda x: x, historic=True))
    plt.xlabel("Time steps")
    plt.ylabel("Max(X)|E(F(X))")
    plt.tight_layout()
    if memetic is None:
        f = figname
    elif memetic:
        f = f"graphs/NEVA_Memetic_Max(X)|E(F(X))_{pb}_N={N}_D={D}"
    else:
        f = f"graphs/NEVA_Max(X)|E(F(X))_{pb}_N={N}_D={D}"
    plt.savefig(f)
    print(f"Figure saved at {f}")
    # for i in V:
    #     plt.plot(d[i])
    # print(max([problem(d) for d in CGA_simple(V, E, k, max_period=max_period, Combine=combination, Mutate=mutate, f=problem, num_steps=num_steps)]))
    # plt.show()
    print("Best found solution:", max([v[num_steps - 1] for v in d]))