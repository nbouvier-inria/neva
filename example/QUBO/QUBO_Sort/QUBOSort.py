import numpy as np

def QUBO_Sort(x: np.ndarray) -> np.ndarray:
    D = x.shape[0]
    n = np.array([[i for i in range(D)]])
    x = x/sum(abs(x))
    I = np.eye(D)
    N = np.tensordot(I, np.transpose(n), axes=0)
    one = np.transpose(np.ones(shape=(1, D)))
    # print("one:", one)
    # print("I:", I)
    Cr = np.tensordot(np.transpose(one), I, axes=0)[0]
    Cc = np.tensordot(I, np.transpose(one), axes=0)
    Cc = np.array([[l[0] for l in j] for j in Cc])
    # print("Cr", Cr,"\nCc: ", Cc)
    R = D * (np.matmul(np.transpose(Cr), Cr) + np.matmul(np.transpose(Cc), Cc))
    # print(N, x, Cr, Cc)
    r = -np.matmul(np.transpose(N), x) 
    r = r - 2 * np.matmul(np.transpose(D*(Cr + Cc)), one)
    return (R, np.transpose(r))

def sparse_to_array(filename: str) -> np.ndarray:
    mat = open(filename, "r")
    n, m = [int(i) for i in mat.readline().split(' ')]
    Q = np.zeros((n,n))
    for _ in range(m):
        i, j, q = [int(i) for i in mat.readline().split(' ')]
        Q[i - 1, j - 1] = q
        Q[j - 1, i - 1] = q
    return Q

T = [464,21,13,5,66,44,123874,112,34,1,22,7]
(R, r) = QUBO_Sort(np.array(T))
print(R)

def QUBO_Value(Q, x):
    return np.matmul(x, np.matmul(R,np.transpose(x))) + np.matmul(r, x)