"""
An example script for solving QUBO using only
one SNN, given that each vertex corresponds to 
a neuron
"""
from lava.proc.monitor.process import Monitor
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.resources import CPU
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.ports import PyInPort, PyOutPort, PyRefPort, PyVarPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort, RefPort, VarPort
import numpy as np
from neva.tools.QUBO_tools import QUBO_Value, sparse_to_array, bound, QUBO_annealing
from matplotlib import pyplot as plt

"""
Parameters
"""
# Q = np.array([[0, 5, 0, -3, 7],
#           [5, 0, -1, 0, 0],
#           [0, -1, 0, -2, 0],
#           [-3, 0, -2, 0, 0],
#           [7, 0, 0, 0, 0]], dtype=int)
Q = sparse_to_array('./benchmarks/gka_sparse_all/gka5c.sparse') # QUBO matrix to solve
N = Q.shape[0]  # Number of neurons = dimension of the problem
dv = 0.1        # Inverse of decay time-constant for voltage decay.
vth = 5         # Neuron threshold voltage, exceeding which, the neuron will spike.
num_steps = 500 # Number of steps to run the algorith for
beta = np.power(1/N, 1/num_steps) # Constant probability of spiking noise decay factor (If set to 1, there is no decay)
minV = None     # mINIMUM MEMBRANE VOLTAGE VALUE

"""
End of parameters
"""


class MyLIF(AbstractProcess):
    """ Specialized QUBO LIF for solving 
    QUBO instances
    ----------
    dv: Inverse of decay time-constant for voltage decay.
    vth: Neuron threshold voltage, exceeding which, the neuron will spike.
    initial_spike: First fired iteration, used to overwrite the neurons with a
    best-found solution
    initial_position : Binary array with a 1 at the neuron position (For N neurons,
    neuron k has an expected array np.eye(1, N, k))
    neighbours : Array containing the weight of all synapses going to a neighbour of
    the neuron
    minV : Minimum value for V
    seed : For now, processes need to be seeded
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.pop("shape", 0)
        dv = kwargs.pop("dv", 0)
        bias_mant = kwargs.pop("bias_mant", 0)
        vth = kwargs.pop("vth", 10)
        initial_position = kwargs.pop("initial_position", 0)
        init_i = 0
        for i in range(shape[0]):
            if initial_position[0][i]:
                init_i = i
        neighbours = kwargs.pop("neighbours", 0)
        initial_spike = kwargs.pop("initial_spike", 0)
        minV = kwargs.pop("minV", -float("inf"))
        seed = kwargs.get("seed", 0)
        if minV is None:
            minV = -float("inf")

        self.shape = shape
        self.a_in = InPort(shape=shape)
        self.seed = Var(shape=(1,), init=seed)
        self.a_out = OutPort(shape=shape)
        self.mediator_out = OutPort(shape=shape)
        self.v = Var(shape=(1, ), init=0)
        self.init_i = Var(shape=(1, ), init=init_i)
        self.minV = Var(shape=(1,), init=minV)
        self.dv = Var(shape=(1,), init=dv)
        self.bias_mant = Var(shape=(1, ), init=bias_mant)
        self.vth = Var(shape=(1,), init=vth)
        self.first_iteration = Var(shape=(1, ), init=True)
        self.initial_spike = Var(shape=shape, init=initial_spike)
        self.neighbours = Var(shape=shape, init=neighbours)
        self.initial_position = Var(shape=shape, init=initial_position)


@implements(proc=MyLIF, protocol=LoihiProtocol)
@requires(CPU)
class PyMyLifModel(PyLoihiProcessModel):
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    v: float = LavaPyType(float, float)
    init_i: int = LavaPyType(int, int)
    bias_mant: float = LavaPyType(float, float)
    dv: float = LavaPyType(float, float)
    vth: float = LavaPyType(float, float)
    minV: float = LavaPyType(float, float)
    seed: int = LavaPyType(int, int)
    first_iteration: bool = LavaPyType(bool, bool)
    initial_position: np.ndarray = LavaPyType(np.ndarray, bool)
    initial_spike: np.ndarray = LavaPyType(np.ndarray, bool)
    neighbours: np.ndarray = LavaPyType(np.ndarray, int)
    mediator_out: PyOutPort= LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)


    def run_spk(self):
        # print(f"ITERATION {self.time_step}")
        gen = np.random.Generator(np.random.PCG64(self.seed))
        self.seed = (self.seed + self.time_step ) % 1000000000

        if self.first_iteration:
            self.first_iteration = False
            a_out = self.initial_spike[self.init_i] * self.neighbours
            self.a_out.send(a_out)
            self.mediator_out.send(self.initial_position)
        else:
            a_in_data = self.a_in.recv()[self.init_i]
            self.v = self.v * (1 - self.dv) + a_in_data + self.bias_mant
            self.v = max(self.v, self.minV)
            a_out = self.v >= self.vth
            if a_out or gen.random() < np.power(beta, self.time_step):
                # print(np.power(beta, self.time_step))
                self.v = 0  # Reset voltage to 0
                mediator_out = self.initial_position
            else:
                mediator_out = np.zeros(shape)
            self.mediator_out.send(mediator_out)
            self.a_out.send(a_out*self.neighbours)


class Mediator(AbstractProcess):
    """
    Given a spiking output from a LIF neural network,
    updates the SpikeGenerator to the current best solution
    """
    def __init__(self, Q: np.ndarray, name) -> None:
        super().__init__(name=name)
        shape = (Q.shape[0], )
        self.Q = Var(shape=Q.shape, init=Q)
        self.best = Var(shape=(1, ), init=-float("inf"))
        self.best_value = Var(shape=shape, init=np.zeros(shape))
        self.shape = shape
        self.first_iteration = Var(shape=(1, ), init=True)
        self.current = Var(shape=(1, ), init=0)
        self.s_in = InPort(shape=shape)


@implements(proc=Mediator, protocol=LoihiProtocol)
@requires(CPU)
class PyMediatorModel(PyLoihiProcessModel):
    """Mediator process model."""
    Q: np.ndarray = LavaPyType(np.ndarray, np.ndarray)
    best: np.ndarray = LavaPyType(np.ndarray, float)
    current: np.ndarray = LavaPyType(np.ndarray, float)
    first_iteration: bool = LavaPyType(bool, bool)
    best_value: np.ndarray = LavaPyType(np.ndarray, float)
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool)
    def run_spk(self) -> None:
        s = self.s_in.recv()
        self.current[0] = QUBO_Value(Q, s)
        if self.current[0] > self.best[0]:
            self.best[0] = self.current[0]
            self.best_value = s


if __name__ == "__main__" :

    print("Creating monitor...")
    moni1 = Monitor()
    moni2 = Monitor()
    shape = (Q.shape[0], )
    run_condition = RunSteps(num_steps=num_steps)
    run_cfg = Loihi1SimCfg()

    init = np.random.random(shape)
    print("Creating neurons...")
    L = [MyLIF(shape=shape, seed=np.random.randint(1000000000), dv=dv, bias_mant=0, vth=vth, neighbours=Q[i], initial_spike=init, minV=minV, initial_position=np.eye(1, shape[0], i)) for i in range(Q.shape[0])]
    print("Creating mediator...")
    m = Mediator(Q=Q, name="medi")
    
    print("Connecting neurons...")
    for i in range(Q.shape[0]):
        L[i].mediator_out.connect(m.s_in)
        for j in range(Q.shape[1]):
            if Q[i, j] != 0:
                L[i].a_out.connect(L[j].a_in)

    moni1.probe(m.best, num_steps=num_steps)
    moni2.probe(m.current, num_steps=num_steps)

    print("Running...")
    L[0].run(run_cfg=run_cfg, condition=run_condition)
    datas = moni1.get_data()
    datas2 = moni2.get_data()
    for l in L: 
        l.stop()


    historique = [i for i in datas["medi"]["best"]]
    plt.plot(historique)
    historique = [i for i in datas2["medi"]["current"]]
    plt.plot(historique)
    print("Best found solution:", datas["medi"]["best"][num_steps - 1])
    print("Best found solution with an annealing equivalent:", QUBO_annealing(Q, 10000, lambda x: x))
    plt.show()
