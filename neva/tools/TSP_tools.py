import numpy as np
import tsplib95
import networkx
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib as mpl


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
"EDGE_DATA_SECTION", "EDGE_WEIGHT_TYPE", "EDGE_WEIGHT_FORMAT", "EDGE_WEIGHT_SECTION", "NODE_COORD_SECTION"]:
        if m[0] == "DIMENSION":
            D = int(m[len(m)-1])
        m = [a.strip(':') for a in mat.readline().strip('\n').split()]
    Q = []
    m = [int(i) for i in m]
    k = 0
    for i in range(D):
        L = []
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

def pmx(parent1, parent2):
    # Select two random crossover points
    crossover_points = np.random.choice(range(1, len(parent1)), 2, replace=False)
    crossover_points.sort()

    # Create offspring
    offspring = np.full(len(parent1), np.nan)

    # Copy the segments between the crossover points from the parents
    offspring[crossover_points[0]:crossover_points[1]] = parent1[crossover_points[0]:crossover_points[1]]

    # Map the remaining elements
    for i in range(len(parent1)):
        if np.isnan(offspring[i]):
            val = parent2[i]
            while np.isin(val, offspring):
                val = parent1[np.where(parent2 == val)][0]
            offspring[i] = val

    return offspring.astype(int)

def tsp_from_tsp(filename):
    # load the tsplib problem
    problem = tsplib95.load(filename)

    # convert into a networkx.Graph
    graph = problem.get_graph()

    # convert into a numpy distance matrix
    Q = nx.to_numpy_matrix(graph)
    return Q

def greedy(initial, Q: np.ndarray):
    """ Greedy search of a TSP solution """
    D = Q.shape[0]
    tour = list(initial)
    tour =  tour if tour != [] else [np.random.randint(D)]
    x = tour[len(tour)-1]
    mask = np.zeros(shape=(D,))
    inf = float("inf") * np.ones(shape=(D,))
    for i in tour:
        mask[i] = 1
    for i in range(D-len(tour)):
        next = Q[x]
        next = np.where(mask, inf, next)
        x = np.argmin(next)
        tour.append(x)
        mask[x] = 1
    return np.array(tour)

def plot_TSP(tour, filename):
    """ Plot the TSP solution given a TSPLIB
    filename and a tour """
    
    problem = tsplib95.load(filename)
    graph = problem.get_graph()
    plt.figure(figsize=(10, 10))
    if isinstance(tour, list):
        cmap = mpl.colormaps['coolwarm']
        colors = cmap(np.linspace(0, 1, len(tour)))
        k = 0
        for t in tour:
            plt.clf()
            plt.title("TSP solution")
            plt.axis("off")
            pos = {i:problem.get_display(i) for i in problem.get_nodes()}
            plt.plot([pos[t[i]+1][0] for i in range(len(t))]+[pos[t[0]+1][0]],
                    [pos[t[i]+1][1] for i in range(len(t))]+[pos[t[0]+1][1]],
                    c=colors[k], linewidth=3)
            nx.draw_networkx_nodes(graph, pos=pos, node_color="red", node_size
                                = 100, edgecolors="black")
            # nx.draw_networkx_edges(graph, pos=pos, edge_color="black")
            nx.draw_networkx_labels(graph, pos=pos, font_size=5)
            plt.pause(0.5)
            k +=1
    else:
        plt.title("TSP solution")
        plt.axis("off")
        pos = {i:problem.get_display(i) for i in problem.get_nodes()}
        plt.plot([pos[tour[i]+1][0] for i in range(len(tour))]+[pos[tour[0]+1][0]],
                [pos[tour[i]+1][1] for i in range(len(tour))]+[pos[tour[0]+1][1]],
                color="blue", linewidth=3)
        nx.draw_networkx_nodes(graph, pos=pos, node_color="red", node_size
                            = 100, edgecolors="black")
        # nx.draw_networkx_edges(graph, pos=pos, edge_color="black")
        nx.draw_networkx_labels(graph, pos=pos, font_size=5)
    plt.savefig("../graphs/tsp.png")
    print("Figure saved at ../graphs/tsp.png")
    plt.show()
    return