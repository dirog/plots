import numpy as np
from utils.utils import Source


def _RD(D : float, P : float) -> float:
    if D <= 0 or P < 0:
        raise ValueError("Invalid argument(s).")

    tmp = np.log2(P / D)
    return 1/2 * np.maximum(0, tmp)


def RD(D : np.ndarray, P : np.ndarray) -> np.ndarray:
    
    func = np.vectorize(_RD)
    return func(D,P)


def _iRD(D : float, Pu : float, Psu : float, eta : float) -> float:
    if D < eta:
        return np.nan
    else:
        tmp = np.log2(Psu**2 / (Pu * (D - eta)))
        return 1/2 * np.maximum(0, tmp)
    
       
def iRD(D : np.ndarray, Pu : np.ndarray, Psu : np.ndarray, eta : float) -> np.ndarray:        
    func = np.vectorize(_iRD)
    return func(D, Pu, Psu, eta)


def _RDD(Ds : float, Du  : float, src : float, indirect : bool):
        def indirect_rd(Ds : float, Du  : float, src : float):
            lhs = _RD(Du, src.Pu)
            rhs = _iRD(Ds, src.Pu, src.Psu, src.eta)
            return np.maximum(lhs, rhs)

        def direct_rd(Ds : float, Du  : float, src : float):
            r1 = Ds / src.Ps
            r2 = Du / src.Pu
            if r1 > r2:
                tmp = r1
                r1 = r2
                r2 = tmp
            
            rho = src.rho
            if rho**2 >= (1 - r2) / (1 - r1):
                rd = 1/2 * np.log2(1 / r1)
            elif rho**2 <= (1 - r1) * (1 - r2):
                rd = 1/2 * np.log2((1 - rho**2) / (r1 * r2))
            elif ( rho**2 > (1 - r1) * (1 - r2) ) and ( rho**2 < (1 - r2) / (1 - r1) ):
                rd = 1/2 * np.log2( (1 - rho**2) / ( r1 * r2 - (rho - np.sqrt( (1 - r1) * (1 - r2) ))**2 ) )

            return np.maximum(0, rd)


        if indirect:
            return indirect_rd(Ds,Du,src)
        else:
            return direct_rd(Ds,Du,src)
        

def RDD(Ds : np.ndarray, Du : np.ndarray, src : Source, indirect : bool) -> np.ndarray:    
    func = np.vectorize(_RDD)
    return func(Ds, Du, src, indirect)