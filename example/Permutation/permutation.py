"""
Adaptation of Solving Constraint Satisfaction Problems with Networks of Spiking Neurons
\r\nZeno Jonke&#x;Zeno Jonke†Stefan Habenschuss&#x;Stefan Habenschuss†Wolfgang Maass*Wolfgang Maass*
"""
from lava.proc.lif.process import LIF
from lava.proc.dense.process import Dense
import numpy as np
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.proc.monitor.process import Monitor
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.resources import CPU
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

G = np.array([[0, 5, 0, 3, 7],
              [5, 0, 1, 0, 19],
              [0, 1, 0, 2, 4],
              [3, 0, 2, 0, 8],
              [7, 19, 4, 8, 0]], dtype=int)

class SpikeGen(AbstractProcess):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            shape = kwargs.get('shape', (2,))
            self.out = OutPort(shape=shape)

@implements(proc=SpikeGen, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyProcModelA(PyLoihiProcessModel):
    out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)

    def run_spk(self):
        sortie = np.zeros(shape=self.out.shape)
        sortie[0] = 1
        data = np.array(sortie)
        self.out.send(data)

def TSP_Solve(G: np.ndarray, Nresting=100, no_revisiting=2, no_duplicate=2, tour=0.2, dv=0.3) -> list:
    # Minimal process with an OutPort
    N = G.shape[0]
    m = max([max(i) for i in G])
    temp = np.zeros(G.shape)
    for i in range(N):
        for j in range(N):
            if G[i, j] != 0:
                temp[i, j] = 10*(m- G[i, j]+1)/m
    G = temp
    Ntot = N + Nresting
    spikes = SpikeGen(shape=(N,))
    lifs = [LIF(shape=(N,), name=f"LIF_{i}", dv=dv) for i in range(N)]
    spikes.out.connect(lifs[0].a_in)
    moni = [Monitor() for _ in range(N)]
    dense = [Dense(weights= tour * G) for _ in range(N)]
    lifs[N-1].s_out.connect(dense[N-1].s_in)
    dense[N-1].a_out.connect(lifs[0].a_in)
    wta = [Dense(weights=-no_duplicate*(np.ones(shape=(N, N))-np.eye(N,N))) for _ in range(N)]
    for t in range(N-1):
        lifs[t].s_out.connect(dense[t].s_in)
        dense[t].a_out.connect(lifs[t+1].a_in)
        for r in range(t+1, N):
            temp = Dense(weights=-no_revisiting * np.eye(N, N))
            lifs[t].s_out.connect(temp.s_in)
            temp.a_out.connect(lifs[r].a_in)
    for t in range(N):
        moni[t].probe(lifs[t].s_out, num_steps=Ntot)
        lifs[t].s_out.connect(wta[t].s_in)
        wta[t].a_out.connect(lifs[t].a_in)
    lifs[0].run(condition=RunSteps(Ntot), run_cfg=Loihi1SimCfg(select_tag="floating_pt"))
    datas = [moni[t].get_data()[f"LIF_{t}"]["s_out"] for t in range(N)]
    lifs[0].stop()
    datas = [datas[i][i+Nresting] for i in range(N)]
    results = []
    for a in datas:
        cities = []
        for i in range(N):
            if a[i] == 1:
                cities.append(i+1)
        results.append(cities)
    print(results)

if __name__ == "__main__":
    TSP_Solve(G)
