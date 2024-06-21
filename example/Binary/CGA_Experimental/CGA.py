"""
Implementation of a Cellular Genetic 
Algorithm for optimization
"""

from lava.proc.dense.process import Dense
from lava.proc.monitor.process import Monitor
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.resources import CPU
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort
import numpy as np
import matplotlib.pyplot as plt
from neva.QUBO_tools import QUBO_Value, sparse_to_array, QUBO_annealing
from typing import Dict, List, Tuple
plt.style.use("fivethirtyeight")


def mutate1(x: np.ndarray, k: int=1):
    """
    Random uniform k bitflips 
    """
    return np.logical_xor(x, (np.random.random(x.shape) <= k/x.shape[0]))

def mutate2(x: np.ndarray, Q: np.ndarray):
    """
    One step of the naive QUBO solving algorithm
    """
    return np.minimum(np.ones(x.shape), np.maximum(np.zeros(x.shape),np.matmul(Q, x)))

def mutate3(x: np.ndarray, Q: np.ndarray, n:int):
    """
    n steps of the naive QUBO solving algorithm
    """
    if n > 0:
        return mutate3(mutate2(x, Q), Q, n-1)
    else:
        return x

def dist(x: np.ndarray, y: np.ndarray):
    D = x.shape[0]
    diff = 0
    for i in range(D):
        if x[i] != y[i]:
            diff +=1
    return diff / D * 100



def ring_one_way(n: int) -> Tuple[List[int], List[Tuple[int, int]]]:
    V = [i for i in range(n)]
    E = [(0,n-1)]
    for i in range(n-1):
        E.append((i, i+1))
    return (V, E)

def grid(n: int) -> Tuple[List[int], List[Tuple[int, int]]]:
    V = [i for i in range(n**2)]
    E = []
    for i in range(n-1):
        E.append((i + n*(n-1), i + 1 + n*(n-1)))
        E.append((n-1 + n*(i), n-1 + n*(i+1)))
        for j in range(n-1):
            E.append((i + n*j, i + n*j +1))
            E.append((i + n*j, i + n*(j +1)))
    return V, E

def torus(n: int) -> Tuple[List[int], List[Tuple[int, int]]]:
    V = [i for i in range(n**2)]
    E = []
    for i in range(n-1):
        E.append((i + n*(n-1), i + 1 + n*(n-1)))
        E.append((i + n*(n-1), i))
        E.append((n-1 + n*(i), n-1 + n*(i+1)))
        E.append((n-1 + n*(i), 1 + n*(i)))
        for j in range(n-1):
            E.append((i + n*j, i + n*j +1))
            E.append((i + n*j, i + n*(j +1)))
    return V, E


"""
Parameters
"""
Q = sparse_to_array('gka_sparse_all/gka4a.sparse')
k = 4
V, E = torus(k)
probe = 0
s = 5                                                # Number of step to wait before combining again                 
D = Q.shape[0]
problem = lambda x: QUBO_Value(Q, x)                   # Binary problem to solve
mutate = lambda x: mutate1(x, k=np.log(Q.shape[0]))                           # Method for mutating
figure = []                                         
num_steps = 5000                                       # Number of steps to run the swarm for
p = 2                                                # Max range for random period augmentation (Should be changed only for higher degree networks)
max_period = 50                                       # Period before the particle starts mutating
f0 = lambda x:  x #mutate3(x, Q, 50)                     # Initialisation of positionning
p_reset = 0.1                                           # Probability of going back to the best found result
"""
End of Parameters
"""


class Particle(AbstractProcess):
    """
    Particle for CGA implementation
    f port: Ports for signaling the efficiency of
    the current solution
    data port: Ports for sending datas to
    neghbours
    start: Initial position for the particle
    shape: Shape of the particle
    stand: Maximum waiting time between combinations
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1,))
        start = kwargs.get("start", 0)
        stand = kwargs.get("stand", 0)
        seed = kwargs.get("seed", 0)
        num = kwargs.get("num", 0)

        self.shape = shape
        self.seed = Var(shape=(1,), init=seed)
        self.f_a_in = InPort(shape=(1,))
        self.f_s_out = OutPort(shape=(1,))
        self.data_a_in = InPort(shape=(1,))
        self.data_s_out = OutPort(shape=(1,))
        self.data_neg_a_in = InPort(shape=(1,))
        self.data_neg_s_out = OutPort(shape=(1,))
        self.data = Var(shape=shape, init=start)
        self.best = Var(shape=shape, init=start)
        self.period = Var(shape=(1,), init=0)
        self.num = Var(shape=(1,), init=num)
        self.tick = Var(shape=(1,), init=0)
        self.stand = Var(shape=(1,), init=np.random.randint(1, stand))
        self.stand_tick = Var(shape=(1,), init=np.random.randint(stand))
        self.first_iteration = Var(shape=(1, ), init=True)
        self.zeros = Var(shape=shape, init=np.zeros(shape=shape))


@implements(proc=Particle, protocol=LoihiProtocol)
@requires(CPU)
class PyParticlefModel(PyLoihiProcessModel):
    f_a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)
    f_s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int, precision=1)
    data_neg_a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)
    data_neg_s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int, precision=1)
    data_a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)
    data_s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int, precision=1)
    data: np.ndarray = LavaPyType(np.ndarray, bool)
    best: np.ndarray = LavaPyType(np.ndarray, bool)
    period: np.ndarray = LavaPyType(np.ndarray, int)
    tick: np.ndarray = LavaPyType(np.ndarray, int)
    stand: np.ndarray = LavaPyType(np.ndarray, int)
    stand_tick: np.ndarray = LavaPyType(np.ndarray, int)
    first_iteration: bool = LavaPyType(bool, bool)
    seed: int = LavaPyType(int, int)
    num: int = LavaPyType(int, int)
    zeros: np.ndarray = LavaPyType(np.ndarray, bool)

    def run_spk(self):
        if self.first_iteration:
            self.f_s_out.send(np.array([1]))
            self.data_s_out.send(np.array([-1]))
            self.data_neg_s_out.send(np.array([-1]))
            self.first_iteration = np.array([False])
        else:
            gen = np.random.Generator(np.random.PCG64(self.seed))
            self.seed = (self.seed + self.time_step ) % 1000000000

            f = self.f_a_in.recv()
            data = self.data_a_in.recv()
            data_neg = self.data_neg_a_in.recv()

            if self.tick <= 0:
                # print(f"Spiking with period {self.period}")
                i = int(gen.random() * self.data.shape[0])
                if self.data[i]:
                    self.data_s_out.send(np.array([i]))
                    self.data_neg_s_out.send(np.array([-1]))
                else:
                    self.data_neg_s_out.send(np.array([i]))
                    self.data_s_out.send(np.array([-1]))
                self.f_s_out.send(np.array([1]))
                self.tick = self.period
                self.period += int(gen.random() * p)
                if self.period > max_period:
                    self.data = mutate(self.data)
                # print(f"Period is now {self.period}")
            else:
                # print("In waiting period")
                self.f_s_out.send(np.array([-1]))
                self.data_s_out.send(np.array([-1]))
                self.data_neg_s_out.send(np.array([-1]))
                self.tick -=1

            if self.stand_tick <= 0 and f == 1: 
                if 0 <= data_neg < self.data.shape[0]:
                    self.data[f] = False
                if 0 <= data < self.data.shape[0]:
                    self.data[data] = True
                self.stand_tick = int(gen.random() * self.stand)
            elif self.stand_tick > 0:
                self.stand_tick -= 1

            b = problem(self.best)
            c = problem(self.data)
            if c > b:
                self.best = self.data
                self.period = 0

            elif gen.random() < p_reset:
                self.data = self.best

        if self.num == 1:
            print(f"{self.time_step} iteration...")




def connect(a: Particle, b:Particle):
    a.f_s_out.connect(b.f_a_in)
    a.data_s_out.connect(b.data_a_in)
    b.f_s_out.connect(a.f_a_in)
    b.data_s_out.connect(a.data_a_in)
    b.data_neg_s_out.connect(a.data_neg_a_in)
    a.data_neg_s_out.connect(b.data_neg_a_in)

def combine1(x: np.ndarray, y: np.ndarray):
    """
    Uniform random combination
    """
    m = np.random.random(x.shape) < 0.5
    return x * m + y * (1-m)

def combine2(x: np.ndarray, y: np.ndarray):
    """
    One point combination
    """
    m = int(np.random.random() * x.shape[0])
    retour = np.zeros(x.shape)
    for i in range(m):
        retour[i] = x[i]
    for i in range(m, x.shape[0]):
        retour[i] = y[i]
    return retour



if __name__ == "__main__":
    run_condition = RunSteps(num_steps=num_steps)
    run_cfg = Loihi1SimCfg()
    m = [Monitor() for v in V]
    particles = [Particle(start=f0(np.random.random(size=(D,)) <= 0.5), shape=(D,), stand=s, seed=np.random.randint(1000000000), num=v) for v in V]
    for (u, v) in E:
        connect(particles[u], particles[v])
    
    [m[v].probe(particles[v].data, num_steps=num_steps) for v in V]
    print("Running...")
    particles[probe].run(run_cfg=run_cfg, condition=run_condition)
    data = [m[v].get_data() for v in V]
    particles[probe].stop()
    print("Stop.")
    plt.figure(figsize=(10,8))
    [plt.plot([problem(i) for i in [data[v][i] for i in data[v]][0]['data']]) for v in V]
    x_final = [[data[v][i] for i in data[v]][0]['data'][num_steps - 1] for v in V]
    print(max([problem(x) for x in x_final]))
    print(np.average([dist(x_final[0], x) for x in x_final]))
    print(QUBO_annealing(Q, num_steps * k**2, lambda x:x))
    plt.show()
    