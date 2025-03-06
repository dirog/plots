from utils.utils import ConverseParams
from utils.rd import RD, iRD, RDD
from utils.entropy import h, h2


def RDs(p : ConverseParams):
    if p.indirect:
        return iRD(p.Ds, p.src.Pu, p.src.Psu, p.src.eta) <= p.R * p.ch.cb
    else:
        return RD(p.Ds, p.src.Ps) <= p.R * p.ch.cb

def RDu(p : ConverseParams):
    return RD(p.Du, p.src.Pu) <= p.R * p.ch.cb

def RDsu(p : ConverseParams):
    return RDD(p.Ds, p.Du, p.src, p.indirect) <= p.R * p.ch.cb

def eqvU(p : ConverseParams):
    return p.eqv.deltaU <= p.R * p.ch.cs + h(p.src.Pu) - RD(p.Du, p.src.Pu)

def eqvS(p : ConverseParams):
    return p.eqv.deltaS <= p.R * p.ch.cs + h(p.src.Ps) - RD(p.Ds, p.src.Ps)

def eqvSU(p : ConverseParams):
    return p.eqv.deltaSU <= p.R * p.ch.cs + h2(p.src.detK) - RDD(p.Ds, p.Du, p.src, p.indirect)

def outer(p : ConverseParams):
    return RDsu(p) * RDs(p) * RDu(p) * eqvU(p) * eqvS(p) * eqvSU(p)