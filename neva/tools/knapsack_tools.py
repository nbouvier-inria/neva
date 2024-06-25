import numpy as np

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
