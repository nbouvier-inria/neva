"""
NEVA algorithm applied to permutations.
"""
from neva.permutation.permutationNonParrallelNeva import nonParrallelNevaPermutation
from neva.tools.TSP_tools import tsp_compute, tsp_from_hcp, tsp_from_atsp, mutate1, pmx, tsp_from_tsp
from neva.tools.CGA_tools import torus, ring_one_way
import numpy as np
import time
import matplotlib.pyplot as plt

# G = np.array([[0, 5, 0, 3, 7],
#               [5, 0, 1, 0, 19],
#               [0, 1, 0, 2, 4],
#               [3, 0, 2, 0, 8],
#               [7, 19, 4, 8, 0]], dtype=int)

G = tsp_from_tsp("./benchmarks/ALL_tsp/att48.tsp")

# Number of individuals
N = 256
# Interaction graph
V, E =  torus(N)
# Dimension
D = G.shape[0]
# Problem to solve
f = lambda x:-tsp_compute(x, G)
num_steps = 1500
mutate = mutate1
tau_max=1

if __name__ == "__main__":
    begin = time.time()
    print("Running...")
    d = nonParrallelNevaPermutation(
        V=V,
        E=E,
        D=D,
        f=f,
        num_steps=num_steps,
        probe=True,
        tau_max=tau_max,
        Mutate=mutate,
        Combine=pmx
    )
    end = time.time()
    print("Computation time : ", round(end-begin, 3) , "s")
    plt.figure(figsize=(10, 8))
    plt.plot([-max([d[i][j] for i in V]) for j in range(num_steps)])
    # for i  in V:
    #     plt.plot([-d[i][j] for j in range(num_steps)])
    plt.plot([-np.average([d[i][j] for i in V]) for j in range(num_steps)])
    plt.xlabel("Time steps")
    plt.ylabel("Min(f(x))|E(f(x))")
    plt.tight_layout()
    f = f"../graphs/NEVA_permutation"
    plt.show()
    plt.savefig(f)
    print(f"Figure saved at {f}")
    # for i in V:
    #     plt.plot(d[i])
    # print(max([problem(d) for d in CGA_simple(V, E, k, max_period=max_period, Combine=combination, Mutate=mutate, f=problem, num_steps=num_steps)]))
    # plt.show()
    print("Best found solution:", -max([v[num_steps - 1] for v in d]))