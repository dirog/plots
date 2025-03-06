import numpy as np

def h(P : float) -> float:
    if P <= 0:
        raise ValueError("Argument less or equal to zero!")
    return 1/2 * np.log2(2 * np.pi * np.e * P)

def h2(detK : float) -> float:
    return 1/2 * np.log2((2 * np.pi * np.e)**2 * np.abs(detK))