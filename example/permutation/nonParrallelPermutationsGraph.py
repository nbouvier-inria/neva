"""
NEVA algorithm applied to permutations.
This example shows the evolution of best
and average solution during the optimization
"""
from neva.permutation.permutationNonParrallelNeva import nonParrallelNevaPermutation
from neva.tools.TSP_tools import tsp_compute, tsp_from_hcp, tsp_from_atsp, mutate1, pmx, tsp_from_tsp, greedy, plot_TSP
from neva.tools.CGA_tools import torus, ring_one_way
import numpy as np
import time
import matplotlib.pyplot as plt


# G = np.array([[0, 5, 0, 3, 7],
#               [5, 0, 1, 0, 19],
#               [0, 1, 0, 2, 4],
#               [3, 0, 2, 0, 8],
#               [7, 19, 4, 8, 0]], dtype=int)

file = "./benchmarks/ALL_tsp/bier127.tsp"
print("Loading graph")
G = tsp_from_tsp(file)
print("Graph loaded")


# Number of individuals
N = 64
# Interaction graph
V, E =  torus(N)
# Dimension
D = G.shape[0]
# Problem to solve
f = lambda x: -tsp_compute(x, G)
# Memetic approach
meme = lambda x: greedy(x, G)
p_meme = 1
# Number of "generations"
num_steps = 500000
# Mutate function
mutate = mutate1
# Crossover function
combine = pmx
# Maximum delay between each exchange, relative to D
tau_max = 3
# Graph set to true if a visual result is needed
graph = False

if __name__ == "__main__":
    begin = time.time()
    print("Running...")
    d = nonParrallelNevaPermutation(
        V=V,
        E=E,
        D=D,
        f=f,
        num_steps=num_steps,
        probe=(not graph),
        tau_max=tau_max,
        Mutate=mutate,
        Combine=combine,
        Meme=meme,
        p_meme=p_meme
    )
    end = time.time()
    print("Computation time : ", round(end-begin, 3) , "s")
    if not graph:
        plt.figure(figsize=(10, 8))
        plt.plot([-max([d[i][j] for i in V]) for j in range(num_steps)])
        # for i  in V:
        #     plt.plot([-d[i][j] for j in range(num_steps)])
        plt.plot([-np.average([d[i][j] for i in V]) for j in range(num_steps)])
        plt.xlabel("Time steps")
        plt.ylabel("Min(f(x))|E(f(x))")
        plt.tight_layout()
        f = f"../graphs/permutNEVA"
        plt.show()
        plt.savefig(f)
        print(f"Figure saved at {f}")
        print("Best found solution:", -max([v[num_steps - 1] for v in d]))
    else:
        best = d[np.argmin([-f(x) for x in d])]
        print("Best found solution:", -f(best))
        d.sort(key=f)
        plot_TSP(tour=d, filename=file)
    # for i in V:
    #     plt.plot(d[i])
    # print(max([problem(d) for d in CGA_simple(V, E, k, max_period=max_period, Combine=combination, Mutate=mutate, f=problem, num_steps=num_steps)]))
    # plt.show()
    