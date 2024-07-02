"""
An implementation of QUBO solving with SNNs through
simulated annealing.
Each explorations, a batch of SNNs are sent to explore
possible neighbours, and their explored solutions are
processed through a temperature function that selects
wich solutions should be chosen to be the starts of the
next batch.
Once temperature is low enough, the best found solution
is returned, and compared to other classical solutions
to compute QUBO.
For more information, see :
Neuromorphic Swarm on RRAM Compute-in-Memory Processor
for Solving QUBO Problem
Ashwin Sanjay Lele et al.
"""

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
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort
import numpy as np
from neva.tools.QUBO_tools import QUBO_Value, sparse_to_array, bound, QUBO_random_solver, QUBO_annealing
from math import exp, log
from random import choice
import matplotlib.pyplot as plt
plt.style.use('_mpl-gallery')

"""
List of parameters
"""

# QUBO problem to solve
Q = sparse_to_array('./benchmarks/gka_sparse_all/gka3f.sparse')
N = Q.shape[0]

# Number of steps computed on the SNNs
num_steps = 30

# Number of neighbourhood explorations
num_explos = int(log(N))

# Number of SNN that runs in parrallel
num_SNN = int(log(N))

# Threshold for the neuron to send out a spike
Vth = 1

# Leakage term
Dv = 0.1

# Uniform combination of result
combination = True

# Temperature for choosing potential exploration points
T = [1-i/num_explos for i in range(num_explos)]

# Probability of giving out spikes based on iteration
E = [1/((i+1)) for i in range(num_explos)]

# Spiking efficiency. Needs to be put in comparison with
# the highest/average weighted degree of Q's vertices
P = [1000 for i in range(num_explos)]

# Constant decay of spike probability during exploration
beta = 0.99

# Minimal value for V, set to None by default
minV = None

"""
End of parameters
"""

class SpikeGenerator(AbstractProcess):
    """
    Spike generator for QUBO solving, generating spikes with
    probability p = spiking_prob*beta^t
    Parameters
    ----------
    spike_prob: Initial probability of spiking
    beta: Constant decay factor
    init_spike: First spikes configuration
    """
    def __init__(self, shape: tuple, spike_prob: float=1, beta: float=0.99, seed=1234, **kwargs) -> None:
        super().__init__()
        init_spike = kwargs.pop("init_spike", np.zeros(shape=shape))
        self.spike_prob = Var(shape=(1, ), init=spike_prob)
        self.s_out = OutPort(shape=shape)
        self.beta: float = Var(shape=(1, ), init=bound(beta, 0, 1))
        self.debut: bool = Var(shape=(1, ), init=True)
        self.gb = Var(shape=shape, init=init_spike)
        self.beta_power_gamma: float = Var(shape=(1, ), init=bound(beta, 0, 1))
        self.seed : int = Var(shape=(1,), init=seed)


@implements(proc=SpikeGenerator, protocol=LoihiProtocol)
@requires(CPU)
class PySpikeGeneratorModel(PyLoihiProcessModel):
    """Spike Generator process model."""
    spike_prob: int = LavaPyType(float, float)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    beta: float = LavaPyType(float, float)
    beta_power_gamma: float = LavaPyType(float, float)
    debut: bool = LavaPyType(bool, bool)
    gb: np.ndarray = LavaPyType(np.ndarray, np.ndarray)
    seed : int = LavaPyType(int, int)

    def run_spk(self) -> None:
        generator = np.random.Generator(np.random.PCG64(self.seed))
        self.seed += (self.seed*np.random.randint(1000) + 1)%21349
        # Generate random spike data
        self.beta_power_gamma *= self.beta
        spike_data = generator.random(size=self.s_out.shape) <= (self.spike_prob * self.beta_power_gamma)
        if self.debut:
            self.debut = False
            spike_data = np.logical_or(spike_data, self.gb)
            self.beta_power_gamma = self.beta
        self.s_out.send(spike_data)


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
        shape = kwargs.get("shape", (1,))
        du = kwargs.pop("du", 0)
        dv = kwargs.pop("dv", 0)
        bias_mant = kwargs.pop("bias_mant", 0)
        vth = kwargs.pop("vth", 10)
        initial_spike = kwargs.pop("initial_spike", 0)
        minV = kwargs.pop("minV", -float("inf"))
        if minV is None:
            minV = -float("inf")

        self.shape = shape
        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)
        self.u = Var(shape=shape, init=0)
        self.v = Var(shape=shape, init=0)
        self.minV = Var(shape=(1,), init=minV)
        self.du = Var(shape=(1,), init=du)
        self.dv = Var(shape=(1,), init=dv)
        self.bias_mant = Var(shape=shape, init=bias_mant)
        self.vth = Var(shape=(1,), init=vth)
        self.first_iteration = Var(shape=(1, ), init=True)
        self.initial_spike = Var(shape=shape, init=initial_spike)


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
    minV: float = LavaPyType(float, float)
    first_iteration: bool = LavaPyType(bool, bool)
    initial_spike: np.ndarray = LavaPyType(np.ndarray, bool)

    def run_spk(self):
        
        if self.first_iteration:
            self.first_iteration = False
            self.s_out.send(self.initial_spike)
        else:
            a_in_data = self.a_in.recv()
            self.u[:] = self.u * (1 - self.du)
            self.u[:] += a_in_data
            self.v[:] = self.v * (1 - self.dv) + self.u + self.bias_mant
            for i in range(self.v.shape[0]):
                self.v[i] = max(self.v[i], self.minV)
            s_out = self.v >= self.vth
            self.v[s_out] = 0  # Reset voltage to 0
            self.s_out.send(s_out)

if __name__ == "__main__":

    x = [np.linspace(num_steps*i, num_steps*(i+1), num_steps) for i in range(num_explos)]

    run_condition = RunSteps(num_steps=num_steps)

    run_cfg = Loihi1SimCfg(select_tag="floating_pt")

    maxi = -float("inf")
    gb = np.zeros(shape=Q.shape[0])
    t = [0 for _ in range(num_SNN)]
    t_formation = [gb for _ in range(num_SNN)]


    for k in range(num_explos):

        # Finding a new set of SNNs based on current temperature
        # and neighborood found at previous iteration
        set = []
        if combination:
            masque = np.random.random((N, )) <= 0.5
            j = np.random.random((N, )) <= 0.5
            for i in t:
                set.append(j*masque + i*(1-masque))
        else:
            perm = np.random.permutation(len(t))
            j = 0
            while len(set) < num_SNN and j < len(t):
                score = t[perm[j]]
                if score >= maxi or np.random.random() < exp(-(maxi - score)/(T[k]*maxi)):
                    set.append(t_formation[perm[j]])
                j += 1
            while len(set) < num_SNN:
                set.append(choice(t_formation))

        # Generating the SNNs
        lif = [MyLIF(shape=(Q.shape[0], ), vth=Vth,                 # Spike if V >= Vth
                    dv=0,                                          # V(t+1) = V(t)*(1-dv)
                    minV=minV,
                    du=1, bias_mant=0., name='lif', 
                    initial_spike=set[i]                           # Initial configuration chosen
                    ) for i in range(num_SNN)]
        dense = [Dense(shape=(Q.shape[0], ), weights=Q, name='dense') for _ in range(num_SNN)]
        spikes = [SpikeGenerator(shape=(Q.shape[0], ),
                                spike_prob=E[k],                   # Current probability of spiking
                                beta=beta, seed=i+np.random.randint(0,1000)) for i in range(num_SNN)]
        spikes_synapse = [Dense(shape=(Q.shape[0], ), 
                                weights=np.eye(N=Q.shape[0])*P[k],  # One to one connection with the
                                                                    # vertexes 
                                name='dense') for _ in range(num_SNN)]

        # Connecting the SNNs
        for i in range(num_SNN):
            spikes[i].s_out.connect(spikes_synapse[i].s_in)
            spikes_synapse[i].a_out.connect(lif[i].a_in)
            lif[i].s_out.connect(dense[i].s_in)
            dense[i].a_out.connect(lif[i].a_in)

        # Generating Monitors
        monitor = [Monitor() for _ in range(num_SNN)]
        for i in range(num_SNN):
            monitor[i].probe(lif[i].s_out, num_steps=num_steps)

        # Starting the simulation
        [lif[i].run(condition=run_condition, run_cfg=run_cfg) for i in range(num_SNN)]

        # Retrieving datas
        datas = [monitor[j].get_data() for j in range(num_SNN)]

        # Analizing datas based to get a large neighborhood
        # from previous starting points used
        t = []
        t_formation =  []
        for d in datas:
            if combination :
                t.append(QUBO_Value(Q, d["lif"]["s_out"][num_steps-1]))
                t_formation.append(d["lif"]["s_out"][num_steps-1])
            else:
                for i in range(num_steps):
                    t.append(QUBO_Value(Q, d["lif"]["s_out"][i]))
                    t_formation.append(d["lif"]["s_out"][i])

        m = np.argmax(t)

        if t[m] > maxi:
            maxi = t[m]
            gb = t_formation[m]
        

        print("The maximum found at step", k+1, "is", maxi)

        [lif[j].stop() for j in range(num_SNN)]


    print("The maximum found with a random equivalent is", QUBO_random_solver(Q, num_explos*num_SNN*num_steps*N))

    # Possible temperature function :  lambda x : x, lanbda x : x**2, lambda x : log(1+x)/log(2)
    print("The maximum found with an annealing equivalent is", QUBO_annealing(Q, num_explos*num_SNN*num_steps*N, lambda x : x))