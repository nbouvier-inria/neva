import numpy as np

def tsp_compute(x:np.ndarray, Q:np.ndarray):
    tot = 0
    i = x[x.shape[0] - 1]
    for j in x:
        tot += Q[i, j]
        i = j
    return tot

def tsp_from_hcp(filename):
    mat = open(filename, "r")
    m = (mat.readline().strip('\n').split())
    while m[0] in ["NAME", "COMMENT", "DIMENSION", "TYPE", "EDGE_DATA_FORMAT", "EDGE_LIST",
"EDGE_DATA_SECTION"]:
        if m[0] == "DIMENSION":
            D = int(m[2])
        m = (mat.readline().strip('\n').split())
    Q = 2 * np.ones((D, D))
    m = [int(i) for i in m]
    while m[0] != -1:
        i, j= m
        Q[i - 1, j - 1] = 1
        Q[j - 1, i - 1] = 1
        m = [int(i) for i in mat.readline().strip('\n').split()]
    return Q

def tsp_from_atsp(filename):
    mat = open(filename, "r")
    m = [a.strip(':') for a in mat.readline().strip('\n').split()]
    while m[0] in ["NAME", "COMMENT", "DIMENSION", "TYPE", "EDGE_DATA_FORMAT", "EDGE_LIST",
"EDGE_DATA_SECTION", "EDGE_WEIGHT_TYPE", "EDGE_WEIGHT_FORMAT", "EDGE_WEIGHT_SECTION"]:
        if m[0] == "DIMENSION":
            D = int(m[len(m)-1])
        m = [a.strip(':') for a in mat.readline().strip('\n').split()]
    Q = []
    m = [int(i) for i in m]
    for i in range(D):
        L = []
        k = 0
        for j in range(D):
            if k < len(m):
                L.append(m[k])
                k += 1
            else:
                m = [int(a.strip(':')) for a in mat.readline().strip('\n').split()]
                L.append(m[0])
                k = 1
        Q.append(L)
    Q = np.array(Q)
    return Q

def mutate1(x: np.ndarray):
    n = x.shape[0]
    i = np.random.randint(n)
    j = np.random.randint(n)
    temp = x[i]
    x[i] = x[j]
    x[j] = temp
    return x