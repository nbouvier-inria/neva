from neva.tools.QUBO_tools import simulated_annealing, sparse_to_array, QUBO_Value
from neva.binary.efficient_neva import efficient_neva, torus, combine1, mutate1, mutate3, ring_one_way, grid
import numpy as np
from neva.tools.SAT_Tools import cnf_to_sat, evaluate
"""
Parameters
"""

pb= "gka2d"
# Q = sparse_to_array(f"gka_sparse_all/{pb}.sparse")  # Import benchmark instances as a numpy array
sat = cnf_to_sat(f"SAT/uf200-860/uf200-06.cnf")
N = 25
V, E =  torus(N) # ring_one_way(N)
config = "Torus"
probe = 0
s = 5  # Number of step to wait before combining again
D = 200 # Q.shape[0]
memetic = False
problem = lambda x: evaluate(sat, x) # QUBO_Value(Q, x)  # Binary problem to solve
combination = lambda x, y: combine1(x, y)  # Method for combining solutions
gamma = np.log(D)/D
mutate = lambda x: mutate1(x, k=D*gamma) if not memetic else (mutate1(x, k=D*gamma) if np.random.random() >= 0.5 else mutate3(x, Q, 1))  # simple_annealing(Q, 200, temperature=lambda x:x**2, s=x)  # Method for mutating
num_steps = 1000 # Number of steps to run the swarm for
k = 4  # Maximum waiting time
max_period = 10 # Period before the particle starts mutating
test_cases = 20 # Number of test cases
"""
End of Parameters
"""

for j in range(1):
    datas = []
    sa = []
    for i in range(test_cases):
        print("Running test case", i)
        d = efficient_neva(
                V,
                E,
                k,
                max_period=max_period,
                Combine=combination,
                Mutate=mutate,
                f=problem,
                num_steps=num_steps,
                probe=False,
                D=D
            )
        sa.append(simulated_annealing(problem, D, num_steps, lambda x: x))
        m = max([problem(d) for d in d])
        datas.append(m)
print(f"{round(np.average(datas))}$\pm${round(np.std(datas))} & {round(np.average(sa))}$\pm${round(np.std(sa))} \\\\") # & {round(np.average(sa), 1)}$\pm${round(np.std(sa), 1)}")

    

