import numpy as np
from math import cos, pi
import benchmarksbis as bs

def weierstrass(x: np.ndarray, x0: np.ndarray):
    w = bs.Weierstrass(shift=x0)
    return w(x)["obj"]

def quadratic(x: np.ndarray, x0: np.ndarray):
    return sum([i**2 for i in (x + x0)])

def rastrigin(x: np.ndarray, x0: np.ndarray):
    n = x.shape[0]
    return (10 * n + sum([i**2 - 10 * cos(2 * pi * i) for i in (x - x0)]))

def cigar(x: np.ndarray, x0: np.ndarray):
    return (x-x0)[0] ** 2 + (10**6) * np.sum(np.power((x-x0)[1:], 2))

def happycat(x: np.ndarray, x0: np.ndarray):
    z = x - x0
    znorm = np.sum(z**2)
    return ((znorm - len(z)) ** 2) ** 1/8 + (0.5 * znorm + np.sum(z)) / len(z)+ 0.5

def levy(x: np.ndarray, x0: np.ndarray):
    y = x - x0
    return np.sin(np.pi * y[0]) ** 2+ np.sum((y[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * y[:-1] + 1) ** 2))  + (y[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * y[-1]) ** 2)