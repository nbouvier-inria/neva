from lava.proc.lif.process import LIF
from lava.proc.dense.process import Dense
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
from QUBOSort import QUBO_Value, QUBO_Sort, T
from matplotlib import pyplot as plt


class MyLIF(AbstractProcess):
    """Leaky-Integrate-and-Fire (LIF) neural Process.
    LIF dynamics abstracts to:
    u[t] = u[t-1] * (1-du) + a_in              # neuron current
    v[t] = v[t-1] * (1-dv) + u[t] + bias_mant  # neuron voltage
    s_out = v[t] > vth                         # spike if threshold is exceeded
    v[t] = 0                                   # reset at spike
    Parameters
    ----------
    du: Inverse of decay time-constant for current decay.
    dv: Inverse of decay time-constant for voltage decay.
    bias: Neuron bias.
    vth: Neuron threshold voltage, exceeding which, the neuron will spike.
    initial_spike: First fired iteration, used to overwrite the neurons with a
    best-found solution
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
        if minV is None:
            minV = -float("inf")

        self.shape = shape
        self.a_in = InPort(shape=shape)
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
    first_iteration: bool = LavaPyType(bool, bool)
    initial_position: np.ndarray = LavaPyType(np.ndarray, bool)
    initial_spike: np.ndarray = LavaPyType(np.ndarray, bool)
    neighbours: np.ndarray = LavaPyType(np.ndarray, int)
    mediator_out: PyOutPort= LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)


    def run_spk(self):
        # print(f"ITERATION {self.time_step}")
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
            if a_out or np.random.random() < np.power(beta, self.time_step):
                print(np.power(beta, self.time_step))
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
        self.current[0] = QUBO_Value(Q, r, s)
        if self.current[0] > self.best[0]:
            self.best[0] = self.current[0]
            self.best_value = s


if __name__ == "__main__" :

    """
    Parameters
    """
    # Q = np.array([[0, 5, 0, -3, 7],
    #           [5, 0, -1, 0, 0],
    #           [0, -1, 0, -2, 0],
    #           [-3, 0, -2, 0, 0],
    #           [7, 0, 0, 0, 0]], dtype=int)
    (Q, r) = QUBO_Sort(np.array(T))
    N = Q.shape[0]
    dv = 0
    vth = 1
    num_steps = 500
    beta = np.power(1/N, 1/num_steps)
    minV = None

    """
    End of parameters
    """
    print("Creating monitor...")
    moni1 = Monitor()
    moni2 = Monitor()
    shape = (Q.shape[0], )
    run_condition = RunSteps(num_steps=num_steps)
    run_cfg = Loihi1SimCfg()

    init = np.random.random(shape)
    print("Creating neurons...")
    L = [MyLIF(shape=shape, dv=dv, bias_mant=0, vth=vth, neighbours=Q[i], initial_spike=init, minV=minV, initial_position=np.eye(1, shape[0], i)) for i in range(Q.shape[0])]
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

    L[0].run(run_cfg=run_cfg, condition=run_condition)
    datas = moni1.get_data()
    datas2 = moni2.get_data()
    for l in L: 
        l.stop()


    historique = [i for i in datas["medi"]["best"]]
    historique = [i for i in datas2["medi"]["current"]]
    print(datas["medi"]["best"][num_steps - 1])