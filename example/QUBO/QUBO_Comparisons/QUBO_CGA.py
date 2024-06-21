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

def CGA(Q, num_steps=10, p=3, max_period=15, p_err=0):
    def mutate1(x: np.ndarray, k: int=1):
        """
        Random uniform k bitflips 
        """
        return np.logical_xor(x, (np.random.random(x.shape) <= k/x.shape[0]))

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

    """
    Parameters
    """
    D = Q.shape[0]
    k = int(np.log(D))+1 
    V, E = grid(k)
    probe = 0
    s = 5                                                # Number of step to wait before combining again                 
    
    problem = lambda x: QUBO_Value(Q, x)                   # Binary problem to solve
    combination = lambda x, y: combine2(x, y)              # Method for combining solutions
    mutate = lambda x: mutate1(x, k=5)                           # Method for mutating            
                                                   # Max range for random period augmentation (Should be changed only for higher degree networks)
    max_period = 15                                       # Period before the particle starts mutating
    f0 = lambda x:  x #mutate3(x, Q, 50)                     # Initialisation of positionning
    p_err = 0                                           # Probability of combining even if the rsult will be less
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
            self.data_a_in = InPort(shape=shape)
            self.data_s_out = OutPort(shape=shape)
            self.data = Var(shape=shape, init=start)
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
        f_s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)
        data_a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)
        data_s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)
        data: np.ndarray = LavaPyType(np.ndarray, bool)
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
                self.f_s_out.send(np.array([False]))
                self.data_s_out.send(self.data)
                self.first_iteration = np.array([False])
            else:
                gen = np.random.Generator(np.random.PCG64(self.seed))
                self.seed = (self.seed + self.time_step ) % 1000000000
                f = self.f_a_in.recv()
                data = self.data_a_in.recv()

                if self.tick <= 0:
                    # print(f"Spiking with period {self.period}")
                    self.f_s_out.send(np.array([True]))
                    self.data_s_out.send(self.data)
                    self.tick = self.period
                    self.period += int(gen.random() * p)
                    if self.period > max_period:
                        self.data = mutate(self.data)
                    # print(f"Period is now {self.period}")
                else:
                    # print("In waiting period")
                    self.f_s_out.send(np.array([False]))
                    self.data_s_out.send(self.zeros)
                    self.tick -=1

                if self.stand_tick <= 0 and f == 1: # and problem(data) >= problem(self.data): # TODO: Choose the data to combine with probalistically
                    # print("Combining")
                    new = combination(data, self.data)
                    if problem(new) > problem(self.data):
                        self.data = new
                        self.period = 0
                    elif gen.random() < p_err:
                        self.data = new
                    self.stand_tick = int(gen.random() * self.stand)
                elif self.stand_tick > 0:
                    self.stand_tick -= 1

                if self.num == 1:
                    print(f"{self.time_step} iteration...")




    def connect(a: Particle, b:Particle):
        a.f_s_out.connect(b.f_a_in)
        a.data_s_out.connect(b.data_a_in)
        b.f_s_out.connect(a.f_a_in)
        b.data_s_out.connect(a.data_a_in)

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



    run_condition = RunSteps(num_steps=num_steps)
    run_cfg = Loihi1SimCfg()
    m = [Monitor() for v in V]
    particles = [Particle(start=f0(np.random.random(size=(D,)) <= 0.5), shape=(D,), stand=s, seed=np.random.randint(1000000000), num=v) for v in V]
    for (u, v) in E:
        connect(particles[u], particles[v])
    
    [m[v].probe(particles[v].data, num_steps=num_steps) for v in V]
    print("Running CGA...")
    particles[probe].run(run_cfg=run_cfg, condition=run_condition)
    data = [m[v].get_data() for v in V[:1]]
    particles[probe].stop()
    print("Stop.")
    x_final = [[data[v][i] for i in data[v]][0]['data'] for v in V[:1]]
    return(max([problem(x) for x in x_final[0]]))

if __name__ == "__main__":
    print(CGA(Q = sparse_to_array('gka_sparse_all/gka4a.sparse')))