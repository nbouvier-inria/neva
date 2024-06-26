import numpy as np
from ortools.algorithms.python import knapsack_solver

def knap_to_problem(filename):
    f = open(filename, "r")
    n, wmax = [int(i) for i in f.readline().split(' ')]
    pb = []
    for _ in range(n):
        v1, w1 = [int(i) for i in f.readline().split(' ')]
        pb.append((v1, w1))
    pb = np.array(pb, dtype=int)
    def aux(x):
        x = np.array(x, dtype=bool)
        value = 0
        weight = 0

        value,weight = pb[x].sum(axis=0)
        #for i in range(n):
        #    if x[i]:
        #        value += int(pb[i][0])
        #        weight += int(pb[i][1])
        
        return value if weight <= wmax else 0
    return aux

def knap_to_problem_penalisation(filename):
    f = open(filename, "r")
    n, wmax = [int(i) for i in f.readline().split(' ')]
    pb = []
    for _ in range(n):
        v1, w1 = [int(i) for i in f.readline().split(' ')]
        pb.append((v1, w1))
    pb = np.array(pb, dtype=int)
    def aux(x):
        x = np.array(x, dtype=bool)
        value = 0
        weight = 0

        value,weight = pb[x].sum(axis=0)
        #for i in range(n):
        #    if x[i]:
        #        value += int(pb[i][0])
        #        weight += int(pb[i][1])
        
        return value if weight <= wmax else value - 1/50*(weight - wmax)**2
    return aux

def knap_solve(filename):
    f = open(filename, "r")
    n, wmax = [int(i) for i in f.readline().split(' ')]
    capacities = [wmax]
    weights = []
    values = []
    for _ in range(n):
        v1, w1 = [int(i) for i in f.readline().split(' ')]
        values.append(v1)
        weights.append(w1)
    solver = knapsack_solver.KnapsackSolver(
        knapsack_solver.SolverType.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER,
        "KnapsackExample",
    )
    solver.init(profits=values, weights=[weights], capacities=capacities)
    return solver.solve()

if __name__ == "__main__":
    pb= "large_scale/knapPI_1_1000_1000_1" 
    filename = f"../instances_01_KP/{pb}"  
    print(knap_solve(filename=filename))