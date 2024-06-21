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
from neva.QUBO_tools import QUBO_Value, sparse_to_array, bound, QUBO_annealing


class SpikeGenerator(AbstractProcess):
    """
    Spike generator for QUBO solving, generating spikes with
    probability spiking_prob*beta^t
    spike_prob: Initial probability of spiking
    beta: Constant decay factor
    """
    def __init__(self, shape: tuple, spike_prob: float=1, beta: float=0.99) -> None:
        super().__init__()
        self.spike_prob = Var(shape=(1, ), init=spike_prob)
        self.beta: float = Var(shape=(1, ), init=bound(beta, 0, 1))

        self.beta_power_gamma: float = Var(shape=(1, ), init=bound(beta, 0, 1))
        self.s_out = OutPort(shape=shape)
        self.reset = InPort(shape=(1, ))
        self.gb = InPort(shape=shape)


@implements(proc=SpikeGenerator, protocol=LoihiProtocol)
@requires(CPU)
class PySpikeGeneratorModel(PyLoihiProcessModel):
    """Spike Generator process model."""
    spike_prob: int = LavaPyType(float, float)
    beta: float = LavaPyType(float, float)

    beta_power_gamma: float = LavaPyType(float, float)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    reset: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool)
    gb: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool)

    def run_spk(self) -> None:
        # Generate random spike data
        
        # print("SpikeGen awaits Mediator")
        r = self.reset.recv()
        gb = self.gb.recv()
        # print("SpikeGen received Mediator")
        if r[0]:
            self.beta_power_gamma = self.beta
        self.beta_power_gamma *= self.beta
        spike_data = np.random.random(size=self.s_out.shape) <= (self.spike_prob * self.beta_power_gamma)
        if r[0]:
            spike_data = np.logical_or(spike_data, gb)
        # print("SpikeGen sends to Synapses")
        self.s_out.send(spike_data)
        # print("SpikeGen sent to Synapses")


class Mediator(AbstractProcess):
    """
    Given a spiking output from a LIF neural network,
    updates the SpikeGenerator to the current best solution
    """
    def __init__(self, Q: np.ndarray) -> None:
        super().__init__()
        shape = (Q.shape[0], )
        self.Q = Var(shape=Q.shape, init=Q)
        self.best = Var(shape=(1, ), init=-float("inf"))
        self.best_value = Var(shape=shape, init=np.zeros(shape))
        self.shape = shape
        self.first_iteration = Var(shape=(1, ), init=True)

        self.reset = OutPort(shape=(1, ))
        self.gb = OutPort(shape=shape)
        self.s_in = InPort(shape=shape)


@implements(proc=Mediator, protocol=LoihiProtocol)
@requires(CPU)
class PyMediatorModel(PyLoihiProcessModel):
    """Mediator process model."""
    Q: np.ndarray = LavaPyType(np.ndarray, np.ndarray)
    best: float = LavaPyType(float, float)
    first_iteration: bool = LavaPyType(bool, bool)
    best_value: np.ndarray = LavaPyType(np.ndarray, float)

    reset = LavaPyType(PyOutPort.VEC_DENSE, bool)
    gb = LavaPyType(PyOutPort.VEC_DENSE, bool)
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool)
    def run_spk(self) -> None:
        if self.first_iteration:
            self.first_iteration = False
            s = np.zeros(self.s_in.shape)
        else:
            s = self.s_in.recv()
        r = np.array([False])
        t = QUBO_Value(Q, s)
        if t > self.best:
            self.best = t
            self.best_value = s
            r = np.array([True])
        self.reset.send(r)
        self.gb.send(s)


class MyDense(AbstractProcess):
    """Dense connections between neurons.
    Realizes the following abstract behavior:
    a_out = W * s_in
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1, 1))
        self.s_in = InPort(shape=(shape[1],))
        self.a_out = OutPort(shape=(shape[0],))
        self.weights = Var(shape=shape, init=kwargs.pop("weights", 0))


@implements(proc=MyDense, protocol=LoihiProtocol)
@requires(CPU)
class PyMyDenseModel(PyLoihiProcessModel):
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool)
    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    weights: np.ndarray = LavaPyType(np.ndarray, float)

    def run_spk(self):
        s_in = np.array(self.s_in.recv(), bool)
        a_out = self.weights[:, s_in].sum(axis=1)
        self.a_out.send(a_out)


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
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1,))
        du = kwargs.pop("du", 0)
        dv = kwargs.pop("dv", 0)
        bias_mant = kwargs.pop("bias_mant", 0)
        vth = kwargs.pop("vth", 10)

        self.shape = shape
        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)
        self.u = Var(shape=shape, init=0)
        self.v = Var(shape=shape, init=0)
        self.du = Var(shape=(1,), init=du)
        self.dv = Var(shape=(1,), init=dv)
        self.bias_mant = Var(shape=shape, init=bias_mant)
        self.vth = Var(shape=(1,), init=vth)
        self.first_iteration = Var(shape=(1, ), init=True)


@implements(proc=MyLIF, protocol=LoihiProtocol)
@requires(CPU)
class PyMyLifModel(PyLoihiProcessModel):
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)
    u: np.ndarray = LavaPyType(np.ndarray, float)
    v: np.ndarray = LavaPyType(np.ndarray, float)
    bias_mant: np.ndarray = LavaPyType(np.ndarray, float)
    du: float = LavaPyType(float, float)
    dv: float = LavaPyType(float, float)
    vth: float = LavaPyType(float, float)
    first_iteration: bool = LavaPyType(bool, bool)

    def run_spk(self):
        
        if self.first_iteration:
            self.first_iteration = False
            self.s_out.send(np.zeros(self.s_out.shape))
        else:
            a_in_data = self.a_in.recv()
            self.u[:] = self.u * (1 - self.du)
            self.u[:] += a_in_data
            self.v[:] = self.v * (1 - self.dv) + self.u + self.bias_mant
            s_out = self.v >= self.vth
            self.v[s_out] = 0  # Reset voltage to 0
            self.s_out.send(s_out)

def Feedback(Q, num_steps = 100):
    """
    Time: O(n*d²)
    Process: O(1) 
    (where e = num_explos, s = num_snn, n = num_steps, d = dimension)
    """
    run_condition = RunSteps(num_steps=num_steps)

    run_cfg = Loihi1SimCfg(select_tag="floating_pt")

    '''
    Q = np.array([[0, 5, 0, -3, 7],
                [5, 0, -1, 0, 0],
                [0, -1, 0, -2, 0],
                [-3, 0, -2, 0, 0],
                [7, 0, 0, 0, 0]], dtype=int)
    '''

    lif = MyLIF(shape=(Q.shape[0], ), vth=1., dv=0.1,du=0.1, bias_mant=0., name='lif')
    dense = MyDense(shape=Q.shape, weights=Q, name="dense")
    spikes = SpikeGenerator(shape=(Q.shape[0], ), spike_prob=0.5, beta=1)
    spikes_synapse = MyDense(shape=Q.shape, weights=np.eye(N=Q.shape[0]), name="spike synapse")
    mediator = Mediator(Q=Q)

    spikes.s_out.connect(spikes_synapse.s_in)
    spikes_synapse.a_out.connect(lif.a_in)
    lif.s_out.connect(dense.s_in)
    dense.a_out.connect(lif.a_in)
    mediator.gb.connect(spikes.gb)
    mediator.reset.connect(spikes.reset)
    lif.s_out.connect(mediator.s_in)



    monitor_score = Monitor()
    monitor_score.probe(mediator.best_value, num_steps=num_steps)

    print("Running one SNN with feedback loop...")
    mediator.run(condition=run_condition, run_cfg=run_cfg)

    datas = monitor_score.get_data()

    mediator.stop()
    print("Stop.")
    x = datas["Process_4"]["best_value"][num_steps-1]
    return(QUBO_Value(Q,x))

