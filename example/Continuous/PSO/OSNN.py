"""
This file implements a swarm search with oscillating
spiking neural networks (OSSNs) to find the maximum
of a given function 'f'. 
"""
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
from math import cos, sin, isclose, pi
import matplotlib.pyplot as plt
from neva.tools.benchmarks import rastrigin, quadratic, cigar, happycat, levy, weierstrass
plt.style.use("fivethirtyeight")
from PSO import PSO

"""
Parameters
"""
N = 10                                                      # Number of particles in the swarm
D = 7                                                       # Problem's dimension
x0 = np.array([(np.random.random()-0.5)*2 for i in range(D)])
f = lambda x: -weierstrass(x, x0)                           # Function to maximise
name = "Weierstrass"
figure = []                                          # phase, maxf, maxfpso

start = np.array([(np.random.random(D)-0.5)*4 for i in range(N)])     # Starting points for the swarm
delta = 1.4                                                 # Damping factor
teta = 32 * (2 * pi) / 360                                  # Rotation in radians

num_steps = 300                                             # Number of steps to run the swarm for

"""
End of Parameters
"""

class OSN(AbstractProcess):
    """
    Abstract model for an oscillating neuron
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        position: np.ndarray = kwargs.pop("position", np.array([True]))
        start = kwargs.pop("start", 0)
        init_i = 0
        for i in range(position.shape[0]):
            if position[i] == 1:
                init_i = i

        self.shape = (1, )
        self.position =  Var(shape=position.shape, init=position)
        self.init_i = Var(shape=(1, ), init=init_i)
        self.s_in = InPort(shape=(1, ))
        self.s_out = OutPort(shape=(1, ))
        self.x_out = OutPort(shape=position.shape)
        self.p_in = InPort(shape=position.shape)
        self.g_in = InPort(shape=position.shape)
        self.q = Var(shape=(1, ), init=0)
        self.y = Var(shape=(1, ), init=0)
        self.x = Var(shape=(1, ), init=start)
        self.v = Var(shape=(1, ), init=0)
        self.Th = Var(shape=(1, ), init=0)
        self.Q = Var(shape=(1, ), init=0)
        self.first_iteration = Var(shape=(1, ), init=True)


@implements(proc=OSN, protocol=LoihiProtocol)
@requires(CPU)
class PyOSNModel(PyLoihiProcessModel):
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)
    x_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    p_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    g_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    q: np.ndarray = LavaPyType(np.ndarray, float)
    y: np.ndarray = LavaPyType(np.ndarray, float)
    x: np.ndarray = LavaPyType(np.ndarray, float)
    v: np.ndarray = LavaPyType(np.ndarray, float)
    Th: np.ndarray = LavaPyType(np.ndarray, float)
    Q: np.ndarray = LavaPyType(np.ndarray, float)
    position: np.ndarray = LavaPyType(np.ndarray, bool)
    init_i: np.ndarray = LavaPyType(np.ndarray, int)
    first_iteration: np.ndarray = LavaPyType(np.ndarray, bool)

    def run_spk(self):
        if not self.first_iteration:
            s_in = self.s_in.recv()
            g = self.g_in.recv()[self.init_i]
            p = self.p_in.recv()[self.init_i]
        else:
            s_in = False
            p = self.x
            g = self.x
            self.first_iteration = False
        self.q = 1/2 * (p + g)
        self.y = self.x - self.q
        self.Th = abs(p - g)
        self.Q = p - self.q
        if abs(self.y) > self.Th:
            self.v = self.v - (self.y - self.Q)
            self.y = self.Q 
            self.s_out.send(np.array([True]))
        elif s_in:
            # if self.init_i == 0:
            #     print(f"Before {self.y}, {self.v}")
            self.v = self.v - (self.y - self.Q)
            self.y = self.Q
            self.s_out.send(np.array([False]))
            # if self.init_i == 0:
            #     print(f"After {self.y}, {self.v}")
        else:
            self.y = delta * ( cos(teta) * self.y + sin(teta) * self.v )
            self.v = delta * ( -sin(teta) * self.y + cos(teta) * self.v )
            self.s_out.send(np.array([False]))
        self.x = self.y + self.q
        self.x_out.send(self.x * self.position)
        # if self.init_i == 0:
        #     print(f"x = {round(self.x[0], 3)}, v = {round(self.v[0], 3)}, Th = {round(self.Th[0], 3)}, Q = {round(self.Q[0], 3)}, p = {round(f(p), 3)}, g = {round(f(g), 3)}, y = {round(self.y[0], 3)}, q = {round(self.q[0], 3)} on {self.time_step}th iteration")


class Particle(AbstractProcess):
    """
    Abstract model for a particle of OSNs
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.pop("shape")
        start = kwargs.pop("start", np.zeros(shape=shape))

        self.shape = shape
        self.g_in = InPort(shape=shape)
        self.g_out = OutPort(shape=shape)
        self.p_out = OutPort(shape=shape)
        self.x_in = OutPort(shape=shape)
        self.p = Var(shape=shape, init=start)
        self.g = Var(shape=shape, init=start)
        self.first_iteration = Var(shape=(1, ), init=True)


@implements(proc=Particle, protocol=LoihiProtocol)
@requires(CPU)
class PyParticleModel(PyLoihiProcessModel):
    g_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    p_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    x_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    g_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    p: np.ndarray = LavaPyType(np.ndarray, float)
    g: np.ndarray = LavaPyType(np.ndarray, float)
    first_iteration: np.ndarray = LavaPyType(np.ndarray, bool)

    def run_spk(self):
        if self.first_iteration:
            new_g = self.g
            new_x = self.p
            self.first_iteration = False
        else:
            new_g = self.g_in.recv()
            new_x = self.x_in.recv()
        # print(f"Received {f(self.p)} as p, {f(self.g)} as g and {f(new_x)} as x at {self.time_step}th iteration")
        if f(new_x) > f(self.p):
            self.p = new_x
        if f(self.p) > f(self.g):
            self.g = self.p
        if f(new_g) > f(self.g):
            self.g = new_g
        self.p_out.send(self.p)
        self.g_out.send(self.g)



if __name__ == "__main__":
    
    moni1 = Monitor()
    moni2 = Monitor()
    X = [Monitor() for _ in range(N)]
    results = Monitor()
    Particles = [Particle(shape=(D,), start=start[i]) for i in range(N)]
    OSNs = [[OSN(position=np.eye(1, D, j)[0], start=start[i][j]) for j in range(D)] for i in range(N)]

    # Connections between the neurons
    for i in range(D):
        OSNs[N-1][i].s_out.connect(OSNs[0][i].s_in)
        for j in range(N-1):
            OSNs[j][i].s_out.connect(OSNs[j+1][i].s_in)
        for j in range(N):
            OSNs[j][i].x_out.connect(Particles[j].x_in)
            Particles[j].g_out.connect(OSNs[j][i].g_in)
            Particles[j].p_out.connect(OSNs[j][i].p_in)
    Particles[N-1].g_out.connect(Particles[0].g_in)
    for i in range(N-1):
        Particles[i].g_out.connect(Particles[i+1].g_in)
    
    run_condition = RunSteps(num_steps=num_steps)
    run_cfg = Loihi1SimCfg()
    moni1.probe(OSNs[0][0].y,num_steps=num_steps)
    moni2.probe(OSNs[0][0].v,num_steps=num_steps)
    for i in range(N):
        X[i].probe(Particles[i].p, num_steps=num_steps)
    results.probe(Particles[0].g,num_steps=num_steps)

    print("Running...")
    Particles[0].run(run_cfg=run_cfg, condition=run_condition)
    data1 = moni1.get_data()
    data2 = moni2.get_data()
    results = results.get_data()
    X = [X[i].get_data()for i in range(N)]
        
    Particles[0].stop()
    print("Simulation ended.")

    data1 = [data1[i] for i in data1][0]['y']
    data2 = [data2[i] for i in data2][0]['v']
    results = [results[i] for i in results][0]['g']
    X = [[x[i] for i in x][0]['p'] for x in X]
    r = [np.average([f(x[i]) for x in X]) for i in range(num_steps)]

    rfinal, fx, fmax = PSO(f, N, D, start, num_steps)

    print(f"The approximate maximum found is {round(f(results[num_steps-1]), 2)} with OSNNs and {round(f(rfinal), 2)} with PSO")

    if "phase" in figure:
        plt.figure(figsize=(10,10))
        plt.plot([i[0] for i in data1], [i[0] for i in data2])

        plt.xlabel("Deviation y")
        plt.ylabel("Deviation speed v")
        plt.tight_layout()
        plt.savefig(f"graphs/OSNN_v|y_{name}_N={N}_D={D}")
        plt.close()
    if "maxf" in figure:
        plt.figure(figsize=(10,8))
        plt.plot(r)
        plt.plot([f(i) for i in results])
        
        plt.xlabel("Number of step")
        plt.ylabel("Max(f(X)) / E(f(X))")
        plt.tight_layout()
        plt.savefig(f"graphs/OSNN_Max(X)|E(F(X))_{name}_N={N}_D={D}")
        plt.close()
    if "maxfpso" in figure:
        plt.figure(figsize=(10,8))
        plt.plot(fx)
        plt.plot(fmax)
        
        plt.xlabel("Number of step")
        plt.ylabel("Max(f(X)) / E(f(X))")
        plt.tight_layout()
        plt.savefig(f"graphs/PSO_Max(X)|E(F(X))_{name}_N={N}_D={D}")
        plt.close()
    if "yoft" in figure:
        plt.figure(figsize=(10,8))
        plt.plot([i[0] for i in data1])

        plt.xlabel("Deviation y")
        plt.ylabel("Time t")
        plt.tight_layout()
        plt.savefig(f"graphs/OSNN_t|y_{name}_N={N}_D={D}")
        plt.close()

