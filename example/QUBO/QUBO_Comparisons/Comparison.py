from QUBO_SA_Parallel import Parrallel
from neva.QUBO_tools import QUBO_annealing, sparse_to_array, QUBO_random_solver
from Determinist_algorithm import deterministic_solve
from QUBO_Internal_Feedback import Feedback
from QUBO_Neuron_Like import Neuron_Like
from math import sqrt
import numpy as np

Q = sparse_to_array('gka_sparse_all/gka5a.sparse')

def QUBO_tries(Q: np.ndarray, n: int):
    """
    Runs the algorithms for time steps equivalent to O(d*n)
    and a
    """
    d = Q.shape[0]
    retours = {}
    retours["Parrallel"] = Parrallel(Q=Q, num_steps=int(sqrt(n)), num_explos=int(sqrt(n)), num_SNN=int(sqrt(d)), beta=0.99)
    retours["Annealing"] = QUBO_annealing(Q=Q, n=int(n/d), temperature=lambda x:x)
    # retours["Neuronal"] = Neuron_Like(Q=Q, num_steps=n*d, beta=np.power(1/d, 1/(n*d)))
    retours["Feedback"] = Feedback(Q=Q, num_steps=int(n/d))
    retours["Deterministic"] = deterministic_solve(Q=Q, n=int(n/d))
    retours["Random"] = QUBO_random_solver(Q=Q, n=int(n/d))
    return retours

if __name__ == "__main__":
    print(QUBO_tries(Q, 10))