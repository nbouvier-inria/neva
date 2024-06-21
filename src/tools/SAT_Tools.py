from pysat.formula import WCNF
from pysat.examples.fm import FM
import numpy as np

def cnf_to_sat(filename):
    f = open(filename, "r")
    l = f.readline().split(' ')
    while l[0][0] == "c":
        l = f.readline().split(' ')
    _, _, v, _,c,_ = l
    sat = []
    for _ in range(int(c)):
        l = f.readline().strip().split(' ')
        x, y, z, _ = l
        c = []
        for d in [x, y, z]:
            if d[0] == "-":
                d = (False, -int(d)-1)
            else:
                d = (True, int(d)-1)
            c.append(d)
        sat.append(c)
    return sat

def cnf_to_result(filename):
    f = open(filename, "r")
    l = f.readline().split(' ')
    while l[0][0] == "c":
        l = f.readline().split(' ')
    _, _, v, _,c,_ = l
    sat = WCNF()
    for _ in range(int(c)):
        l = f.readline().strip().split(' ')
        x, y, z, _ = l
        sat.append(clause=[int(x), int(y), int(z)])
    fm = FM(sat, verbose=0)
    fm.compute()
    return np.array([i > 0 for i in fm.model])

def evaluate(sat, val):
    tot = 0
    for c in sat:
        (bx, x), (by, y), (bz, z) = c
        if (bx == val[x]) or (by == val[y]) or (bz == val[z]):
            tot +=1
    return tot
