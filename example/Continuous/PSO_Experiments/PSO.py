
import numpy as np
from benchmarks import rastrigin, quadratic, pi, cigar, happycat, levy, weierstrass
import matplotlib.pyplot as plt
    # """
    # Parameters
    # """
    # N = 50               # Number of particles in the swarm
    # D = 7              # Problem's dimension
    # x0 = np.array([(np.random.random()-0.5)*2 for i in range(D)])
    # f = lambda x: - weierstrass(x, x0)
    # w = 0.9
    # c1 = 0.1
    # c2 = 1

    # num_steps = 200     # Number of steps to run the swarm for

    # """
    # End of Parameters
    # """
def PSO(f, N, D, x, num_steps, w=0.9, c1=0.1, c2=1):
    """
    Compute a Particle Swarm Optimization on
    f for numsteps steps, with N particle,
    and D dimensions, starting in x
    w: Inertia
    c1: Personal motivation 
    c2: Collaboration
    """
    gs = []
    xs = []
    v = np.array([np.zeros(D) for i in range(N)])
    p = x
    
    for i in range(num_steps):
        g = [p[np.argmax([f(i) for i in p])] for _ in range(N)]
        gs.append(g[0])
        v = w * v + c1 * np.random.random() * (p - x) + c2 * np.random.random() * (g - x)
        x = x + v
        xs.append(np.average([f(i) for i in x]))
        p = np.array([x[i] if f(x[i]) > f(p[i]) else p[i] for i in range(N)])

    return (gs[num_steps-1], xs, [f(i) for i in gs])