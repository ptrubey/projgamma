"""
Implements Variable Bayes classes
"""
import numpy as np
import numpy.typing as npt
from typing import Callable, Self

class Adam(object):
    # Adam Parameters
    eps    : float = 1e-8
    rate   : float
    decay1 : float
    decay2 : float
    iter   : int
    niter  : int

    # Adam Updateables 
    momentum : npt.NDArray[np.float64] # momentum
    sumofsqs : npt.NDArray[np.float64] # sum of squares of past gradients
    theta    : npt.NDArray[np.float64] # parameter set

    # Loss function
    dloss : Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] # function of theta

    def update(self) -> None:
        self.iter += 1
        dloss = self.dloss(theta = self.theta)
        self.momentum[:] = (
            + self.decay1 * self.momentum
            + (1 - self.decay1) * dloss
            )
        self.sumofsqs[:] = (
            + self.decay2 * self.sumofsqs
            + (1 - self.decay2) * dloss * dloss
            )
        mhat = self.momentum / (1 - self.decay1**self.iter)
        shat = self.sumofsqs / (1 - self.decay2**self.iter)
        self.theta -= mhat * self.rate / (np.sqrt(shat) + self.eps)
        return
    
    def specify_dloss(
            self, 
            func : Callable[[npt.NDArray[np.float64]],npt.NDArray[np.float64]],
            ) -> None:
        self.dloss = func
        return

    def initialization(
            self, 
            rate   : float, # Adam Learning Rate
            decay1 : float, # Adam Decay 1
            decay2 : float, # Adam Decay 2
            niter  : int,   # Number of Adam Iterations per sample
            ) -> None:
        self.decay1 = decay1
        self.decay2 = decay2
        self.rate = rate
        self.iter = 0
        self.niter = niter
        self.momentum = np.zeros(self.theta.shape)
        self.sumofsqs = np.zeros(self.theta.shape)
        return

    def optimize(self) -> None:
        for _ in range(self.niter):
            self.update()
        return
    
    def to_dict(self) -> dict:
        out = {
            'rate' : self.rate,
            'decay1' : self.decay1,
            'decay2' : self.decay2,
            'iter'   : self.iter,
            'niter'  : self.niter,
            'momentum' : self.momentum,
            'sumofsqs' : self.sumofsqs,
            'theta'    : self.theta,
            }
        return out
    
    @classmethod
    def from_dict(cls, out : dict) -> Self:
        return cls(**out)

    @classmethod
    def from_meta(
            cls, 
            theta  : npt.NDArray[np.float64], 
            rate   : float = 1e-3,
            decay1 : float = 0.9, 
            decay2 : float = 0.999, 
            niter  : int = 10,
            ) -> Self:
        out = {
            'rate'     : rate,
            'decay1'   : decay1,
            'decay2'   : decay2,
            'iter'     : 0,
            'niter'    : niter,
            'momentum' : np.zeros(theta.shape),
            'sumofsqs' : np.zeros(theta.shape),
            'theta'    : theta,
            }
        return cls.from_dict(out)
        
    def __init__(
            self, 
            theta    : npt.NDArray[np.float64],
            momentum : npt.NDArray[np.float64],
            sumofsqs : npt.NDArray[np.float64],
            rate     : float, 
            decay1   : float, 
            decay2   : float, 
            iter     : int,
            niter    : int,
            ):
        self.theta    = theta
        self.momentum = momentum
        self.sumofsqs = sumofsqs
        self.rate     = rate
        self.decay1   = decay1
        self.decay2   = decay2
        self.iter     = iter
        self.niter    = niter
        return

# EOFzz