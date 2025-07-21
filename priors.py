from typing import NamedTuple
import numpy as np
import numpy.typing as npt

class GammaPrior(NamedTuple):
    a : float
    b : float

class BetaPrior(GammaPrior): pass 

class InvGammaPrior(GammaPrior): pass

class DirichletPrior(NamedTuple):
    a : float

class UniNormalPrior(NamedTuple):
    mu    : float
    sigma : float

class UniLogNormalPrior(UniNormalPrior): pass

class NormalPrior(NamedTuple):
    mu   : npt.NDArray[np.float64] | float
    Scho : npt.NDArray[np.float64] | float
    Sinv : npt.NDArray[np.float64] | float

class LogNormalPrior(NormalPrior): pass

class InvWishartPrior(NamedTuple):
    nu   : int
    psi  : npt.NDArray[np.float64] | float

class GEMPrior(NamedTuple):
    discount      : float
    concentration : float

# EOF