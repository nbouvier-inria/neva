"""
An example scipt for running bulk
tests on the NEVA algorithm
"""

from neva.tools.QUBO_tools import simulated_annealing, sparse_to_array, QUBO_Value
from neva.binary.nonParrallelNeva import nonParrallelNeva
from neva.tools.CGA_tools import torus, combine1, mutate1, mutate3, ring_one_way, grid, ea
from neva.tools.knapsack_tools import knap_to_problem
import numpy as np
from neva.tools.SAT_Tools import cnf_to_sat, evaluate
"""
Parameters
"""

pb= "large_scale/knapPI_1_100_1000_1" 
filename = f"../instances_01_KP/{pb}"  
N = 24 # Number of individuals, must be even
V, E =  torus(N) # ring_one_way(N)
config = "Torus"
s = 5  # Number of step to wait before combining again
D = 100 # Q.shape[0]
problem = knap_to_problem(filename=filename)   # Binary problem to solve
combination = lambda x, y: combine1(x, y)  # Method for combining solutions
gamma = np.log(D)/D
mutate = lambda x: mutate1(x, k=D*gamma)  # Method for mutating
num_steps = 1000 # Number of steps to run the swarm for
k = 4  # Maximum waiting time
max_period = 10 # Period before the particle starts mutating
test_cases = 20 # Number of test cases
f0 = lambda x:np.random.random(size=(len(x),))<0 # Starting point for the search
"""
End of Parameters
"""

for j in range(1):
    datas = []
    sa = []
    for i in range(test_cases):
        print("Running test case", i)
        d = nonParrallelNeva(
                V=V,
                E=E,
                k=k,
                max_period=max_period,
                Combine=combination,
                Mutate=mutate,
                f=problem,
                num_steps=num_steps,
                probe=False,
                D=D,
                f0=f0
            )
        h = ea(problem=problem, D=D, num_steps=int(num_steps), f0=f0,N=N, cxpb=0.5, mutpb=0.5, mu=int(0.5*N), lambda_=int(0.5*N))
        sa.append(max(h))
        # print(h)
        m = max([problem(d) for d in d])
        datas.append(m)
print(f"{round(np.average(datas))}$\pm${round(np.std(datas))}, {round(np.average(sa))}$\pm${round(np.std(sa))} \\\\") # & {round(np.average(sa), 1)}$\pm${round(np.std(sa), 1)}")

    

