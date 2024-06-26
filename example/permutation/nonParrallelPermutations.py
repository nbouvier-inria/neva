from neva.permutation.permutationNonParrallelNeva import nonParrallelNevaPermutation
from neva.tools.TSP_tools import tsp_compute, tsp_from_hcp
from neva.tools.CGA_tools import torus
import numpy as np
import time
import matplotlib.pyplot as plt

# G = np.array([[0, 5, 0, 3, 7],
#               [5, 0, 1, 0, 19],
#               [0, 1, 0, 2, 4],
#               [3, 0, 2, 0, 8],
#               [7, 19, 4, 8, 0]], dtype=int)

G = tsp_from_hcp("../tsphcp/SSP_1011.hcp")

# Number of individuals
N = 32
# Interaction graph
V, E =  torus(N)
# Dimension
D = G.shape[0]
# Problem to solve
f = lambda x:-tsp_compute(x, G)
num_steps = 100000

if __name__ == "__main__":
    begin = time.time()
    print("Running...")
    d = nonParrallelNevaPermutation(
        V=V,
        E=E,
        D=D,
        f=f,
        num_steps=num_steps,
        probe=True
    )
    end = time.time()
    print("Computation time : ", round(end-begin, 3) , "s")
    plt.figure(figsize=(10, 8))
    plt.plot([-max([d[i][j] for i in V]) for j in range(num_steps)])
    plt.plot([-np.average([d[i][j] for i in V]) for j in range(num_steps)])
    plt.xlabel("Time steps")
    plt.ylabel("Min(f(x))|E(f(x))")
    plt.tight_layout()
    f = f"../graphs/NEVA_permutation"
    plt.savefig(f)
    print(f"Figure saved at {f}")
    # for i in V:
    #     plt.plot(d[i])
    # print(max([problem(d) for d in CGA_simple(V, E, k, max_period=max_period, Combine=combination, Mutate=mutate, f=problem, num_steps=num_steps)]))
    # plt.show()
    print("Best found solution:", -max([v[num_steps - 1] for v in d]))