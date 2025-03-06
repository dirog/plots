import numpy as np


class Equivocation:
    def __init__(self, deltaS, deltaU, deltaSU) -> None:
        self.deltaS  = deltaS
        self.deltaU  = deltaU
        self.deltaSU = deltaSU


class Source:
    def __init__(self, Ps, Pu, Psu) -> None:
        self.Ps  = Ps
        self.Pu  = Pu
        self.Psu = Psu
        self.rho = Psu / np.sqrt(Ps * Pu)
        assert(self.rho >= 0 and self.rho <= 1)
        self.eta = Ps - Psu**2/Pu
        self.detK = Ps * Pu - Psu*2


class Channel:
    def __init__(self, P, Pn1, Pn2) -> None:
        self.P = P
        self.Pn1 = Pn1
        self.Pn2 = Pn2
        self.cb = 1/2 * np.log2(1 + P / Pn1)
        self.cs = 1/2 * np.log2((P + Pn1) * (Pn1 + Pn2) / (Pn1 * (P + Pn1 + Pn2)))


class ConverseParams:
    def __init__(self, R, Ds, Du, eqv : Equivocation, src : Source, ch : Channel, indirect : bool) -> None:
        self.indirect = indirect
        self.R = R
        self.Ds = Ds
        self.Du = Du
        self.eqv = eqv
        self.src = src
        self.ch = ch


class AuxPower:
    def __init__(self, Pap, Pbp, Pqc, Pqp, Pwc, Ptx) -> None:
        self.Pap = Pap
        self.Pbp = Pbp
        self.Pqc = Pqc
        self.Pqp = Pqp
        self.Pwc = Pwc
        self.Ptx = Ptx
        
    def get_sum(self):
        return self.Pqc + self.Pqp + self.Pwc + self.Ptx


class InnerParams(ConverseParams):
    def __init__(self, converse_params, aux, a1, a2, g):
        super().__init__(converse_params.R, converse_params.Ds, converse_params.Du, converse_params.eqv, converse_params.src, converse_params.ch, indirect=False)
        assert(not converse_params.indirect)
        self.a1 = a1
        self.a2 = a2
        self.g  = g

        self.aux = aux
        assert(self.aux.get_sum() <= self.ch.P)