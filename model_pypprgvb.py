import numpy as np
import numpy.typing as npt
from typing import Self
from numpy.random import choice, gamma, uniform, normal, lognormal
from collections import namedtuple, deque
from scipy.special import digamma

# Custom Modules
from samplers import py_sample_chi_bgsb, py_sample_cluster_bgsb,               \
    StickBreakingSampler, stickbreak, GammaPrior, GEMPrior
from data import euclidean_to_hypercube, Data, Projection
from projgamma import  logd_projgamma_my_mt_inplace_unstable

np.seterr(divide = 'raise', over = 'raise', under = 'ignore', invalid = 'raise')

Prior = namedtuple('Prior','alpha beta chi')

class Samples(object):
    r     : deque[npt.NDArray[np.float64]] | npt.NDArray[np.float64] # radius (projected gamma)
    chi   : deque[npt.NDArray[np.float64]] | npt.NDArray[np.float64] # stick-breaking weights (unnormalized)
    delta : deque[npt.NDArray[np.int32]]   | npt.NDArray[np.int32]   # cluster identifiers
    beta  : deque[npt.NDArray[np.float64]] | npt.NDArray[np.float64] # rate hyperparameter
    alpha : deque[npt.NDArray[np.float64]] | npt.NDArray[np.float64] # shape hyperparameter (inferred)
    zeta  : deque[npt.NDArray[np.float64]] | npt.NDArray[np.float64] # shape parameter (inferred)
    
    def to_dict(self) -> dict:
        out = {
            'r'     : np.stack(self.r),
            'chi'   : np.stack(self.chi),
            'delta' : np.stack(self.delta),
            'zeta'  : np.stack(self.zeta),
            'beta'  : np.stack(self.beta),
            'alpha' : np.stack(self.alpha),
            }
        return out
    
    @classmethod
    def from_dict(cls, out) -> Self:
        return cls(**out)

    @classmethod
    def from_meta(cls, nkeep : int, N : int, S : int, J : int):
        r     = deque([], maxlen = nkeep)
        chi   = deque([], maxlen = nkeep)
        delta = deque([], maxlen = nkeep)
        zeta  = deque([], maxlen = nkeep)
        alpha = deque([], maxlen = nkeep)
        beta  = deque([], maxlen = nkeep)
        r.append(lognormal(mean = 3, sigma = 1, size = N))
        chi.append(1 / np.arange(2, J + 1)[::-1]) # uniform probability
        delta.append(choice(J, N))
        beta.append(gamma(shape = 2, scale = 1 / 2, size = S))
        return cls(r, chi, delta, zeta, alpha, beta)

    def __init__(
            self,
            r     : deque[npt.NDArray[np.float64]] | npt.NDArray[np.float64],
            chi   : deque[npt.NDArray[np.float64]] | npt.NDArray[np.float64],
            delta : deque[npt.NDArray[np.int32]]   | npt.NDArray[np.int32],
            zeta  : deque[npt.NDArray[np.float64]] | npt.NDArray[np.float64],
            alpha : deque[npt.NDArray[np.float64]] | npt.NDArray[np.float64],
            beta  : deque[npt.NDArray[np.float64]] | npt.NDArray[np.float64],
            ):
        self.r     = r
        self.chi   = chi
        self.delta = delta
        self.zeta  = zeta
        self.alpha = alpha
        self.beta  = beta
        return    
    pass

def gradient_resgammagamma_ln(
        theta   : np.ndarray,  # np.stack((mu, tau))    # (2, j, d)
        lYs     : np.ndarray,  # sum of log(Y)          # (j, d)
        n       : np.ndarray,  # number of observations # (j)
        a       : np.ndarray,  # hierarchical shape     # (d) or float
        b       : np.ndarray,  # hierarchical rate      # (d) or float
        ns      : int = 20,
        ) -> npt.NDArray[np.float64]:
    epsilon = normal(size = (ns, *theta.shape[1:]))
    ete     = np.exp(theta[1]) * epsilon
    alpha   = np.exp(theta[0] + ete)

    dtheta = np.zeros((ns, *theta.shape))
    dtheta += lYs
    dtheta -= n.reshape(1,1,-1,1) * digamma(alpha[:,None])
    dtheta += (a - 1) / alpha[:,None]
    dtheta -= b

    dtheta[:,0] *= alpha
    dtheta[:,1] *= alpha * ete
    dtheta[:,0] -= 1.
    dtheta[:,1] -= 1. + theta[1]
    return dtheta.mean(axis = 0)

def gradient_gammagamma_ln(
        theta   : npt.NDArray[np.float64],  # np.stack((mu, tau))
        lYs     : npt.NDArray[np.float64],  # sum of log(Y)
        Ys      : npt.NDArray[np.float64],  # sum of Y
        n       : npt.NDArray[np.int32],  # number of observations
        a       : npt.NDArray[np.float64],  # hierarchical (shape) shape 
        b       : npt.NDArray[np.float64],  # hierarchical (shape) rate
        c       : npt.NDArray[np.float64],  # hierarchical (rate) shape
        d       : npt.NDArray[np.float64],  # hierarchical (rate) rate
        ns      : int = 10,
        ) -> npt.NDArray[np.float64]:
    epsilon = normal(size = (ns, *theta.shape[1:]))
    ete    = np.exp(theta[1]) * epsilon
    alpha  = np.exp(theta[0] + ete)
    N      = n.reshape(1,-1,1)

    dtheta = np.zeros((ns, *theta.shape))
    dtheta += lYs
    dtheta -= N * digamma(alpha[:,None])
    dtheta += (a - 1) / alpha[:, None]
    dtheta -= b
    dtheta += digamma(N * alpha[:,None] + c) * N
    dtheta -= N * np.log(Ys + d)

    dtheta[:,0] *= alpha
    dtheta[:,1] *= alpha * ete
    dtheta[:,0] -= 1
    dtheta[:,1] -= 1 + theta[1]
    
    return dtheta.mean(axis = 0)

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
    dloss : function[npt.NDArray[np.float64]] = None # function of theta

    def update(self) -> None:
        self.iter += 1
        dloss = self.dloss()
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
            func : function[npt.NDArray[np.float64]],
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

class VariationalParameters(object):
    logzeta : Adam
    logalpha : Adam

    def to_dict(self) -> dict:
        out = {
            'logzeta'   : self.logzeta.to_dict(),
            'logaalpha' : self.logalpha.to_dict(),
            }
        return out
    
    @classmethod
    def from_meta(cls, S : int, J : int, **kwargs) -> Self:
        logzeta_mutau = np.zeros((2, J, S))
        logalpha_mutau = np.zeros((2, S))
        logzeta = Adam(logzeta_mutau, **kwargs)
        logalpha = Adam(logalpha_mutau, **kwargs)
        return cls(logzeta = logzeta, logalpha = logalpha)

    @classmethod
    def from_dict(cls, out : dict) -> Self:
        logzeta = Adam.from_dict(out['logzeta'])
        logalpha = Adam.from_dict(out['logalpha'])
        return cls(logzeta = logzeta, logalpha = logalpha)

    def __init__(
            self,
            logzeta : Adam,
            logalpha : Adam,
            ):
        self.logzeta = logzeta,
        self.logalpha = logalpha,
        return
    pass

class Chain(StickBreakingSampler, Projection):
    samples       : Samples
    varparm       : VariationalParameters
    priors        : Prior[GammaPrior,GammaPrior,GEMPrior]
    concentration : float
    discount      : float
    N             : int
    J             : int
    data          : Data
    curr_iter     : int

    @property
    def curr_r(self) -> npt.NDArray[np.float64]:
        return self.samples.r[-1]
    @property
    def curr_chi(self) -> npt.NDArray[np.float64]:
        return self.samples.chi[-1]
    @property
    def curr_delta(self) -> npt.NDArray[np.int32]:
        return self.samples.delta[-1]
    @property
    def curr_alpha(self) -> npt.NDArray[np.float64]:
        return lognormal(
            mean = self.varparm.logalpha.theta[0], 
            sigma = np.exp(self.varparm.logalpha.theta[1])
            )
    @property
    def curr_beta(self) -> npt.NDArray[np.float64]:
        return self.samples.beta[-1]
    @property
    def curr_zeta(self) -> npt.NDArray[np.float64]:
        return lognormal(
            mean = self.varparm.logzeta.theta[0], 
            sigma = np.exp(self.varparm.logzeta.theta[1]),
            )
    
    def update_zeta(
            self, 
            delta : npt.NDArray[np.int32], 
            r     : npt.NDArray[np.float64], 
            alpha : npt.NDArray[np.float64], 
            beta  : npt.NDArray[np.float64],
            ) -> None:
        dmat = delta[:,None] == np.arange(self.J)
        Y = r[:,None] * self.data.Yp
        n = dmat.sum(axis = 0)
        lYs = dmat.T @ np.log(Y) # (np.log(Y).T @ dmat).T
        
        func = lambda: - gradient_resgammagamma_ln(
            self.varparm.logzeta.theta, lYs, n, alpha, beta, self.var_samp,
            )

        self.varparm.logzeta.specify_dloss(func)
        self.varparm.logzeta.optimize()
        
        self.samples.zeta.append(self.curr_zeta)
        return
    
    def update_alpha(
            self, 
            zeta  : npt.NDArray[np.float64], 
            delta : npt.NDArray[np.float64],
            ) -> None:
        active = np.where(np.bincount(delta, minlength = self.J) > 0)[0]
        n = np.array((active.shape[0],))
        lZs = np.log(zeta)[active].sum(axis = 0)
        Zs  = zeta[active].sum(axis = 0)

        func = lambda: - gradient_gammagamma_ln(
            self.varparm.logalpha.theta, 
            lZs, Zs, n,
            *self.priors.alpha,
            *self.priors.beta,
            ns = self.var_samp
            )
                
        self.varparm.logalpha.specify_dloss(func)
        self.varparm.logalpha.optimize()

        self.samples.alpha.append(self.curr_alpha)
        return

    def update_beta(
            self, 
            zeta  : npt.NDArray[np.float64], 
            alpha : npt.NDArray[np.float64], 
            delta : npt.NDArray[np.int32],
            ) -> None:
        active = np.where(np.bincount(delta, minlength = self.J) > 0)[0]
        n = active.shape[0]
        zs = zeta[active].sum(axis = 0)
        As = n * alpha + self.priors.beta.a
        Bs = zs + self.priors.beta.b
        self.samples.beta.append(gamma(shape = As, scale = 1 / Bs))
        return

    def update_r(
            self, 
            zeta  : npt.NDArray[np.float64], 
            delta : npt.NDArray[np.int32]
            ) -> None:
        As = zeta[delta].sum(axis = -1)  # np.einsum('il->i', zeta[delta])
        Bs = self.data.Yp.sum(axis = -1) # np.einsum('il->i', self.data.Yp)
        r = gamma(shape = As, scale = 1 / Bs)
        r[r < 1e-4] = 1e-4               # lower-bounding radius
        self.samples.r.append(r)
        return
    
    def update_chi(self, delta : npt.NDArray[np.int32]) -> None:
        chi = py_sample_chi_bgsb(
            delta, self.discount, self.concentration, self.J,
            )
        self.samples.chi.append(chi)
        return
    
    def update_delta(
            self, 
            zeta : npt.NDArray[np.float64], 
            chi  : npt.NDArray[np.float64],
            ) -> None:
        llk = np.zeros((self.N, self.J))
        logd_projgamma_my_mt_inplace_unstable(
            llk, self.data.Yp, zeta, np.ones(zeta.shape),
            )
        delta = py_sample_cluster_bgsb(chi, llk)
        self.samples.delta.append(delta)
        return

    def iter_sample(self) -> None:
        chi   = self.curr_chi
        alpha = self.curr_alpha
        beta  = self.curr_beta
        zeta  = self.curr_zeta
        self.curr_iter += 1

        self.update_delta(zeta, chi)
        self.update_chi(self.curr_delta)
        self.update_r(zeta, self.curr_delta)
        self.update_zeta(self.curr_delta, self.curr_r, alpha, beta)

        zeta = self.curr_zeta
        self.update_alpha(zeta, self.curr_delta)
        self.update_beta(zeta, self.curr_alpha, self.curr_delta)
        return

    def initialize_sampler(self, nSamp : int) -> None:
        self.samples = Samples(self.gibbs_samp, self.N, self.S, self.J)
        self.varparm = VariationalParameters(
            self.S, self.J, niter = self.var_iter,
            )
        self.curr_iter = 0
        pass
    
    def to_dict(self) -> dict:
        out = {
            'varparm' : self.varparm.to_dict(),
            'samples' : self.samples.to_dict(),
            'data'    : self.data.to_dict(),
            'prior'   : self.priors,
            'time'    : self.time_elapsed_numeric,
            }
        return out
        
    def __init__(
            self, 
            data          : Data, 
            var_samp      : int = 10, 
            var_iter      : int = 10,
            gibbs_samples : int = 1000,
            max_clusters  : int = 200,
            p             : float = 10.,
            prior_alpha   : GammaPrior[float, float] = GammaPrior(1.01, 1.01),
            prior_beta    : GammaPrior[float, float] = GammaPrior(2., 1.),
            prior_chi     : GEMPrior[float, float]   = GEMPrior(0.1,0.1),
            ):
        self.data = data
        assert len(self.data.cats) == 0
        self.N = self.data.nDat
        self.S = self.data.nCol
        self.J = max_clusters
        self.p = p
        self.var_samp = var_samp
        self.var_iter = var_iter
        self.gibbs_samp = gibbs_samples
        self.priors = Prior(
            GammaPrior(*prior_alpha), 
            GammaPrior(*prior_beta), 
            GEMPrior(*prior_chi)
            )
        self.set_projection()
        return

class Result(object):
    samples              : Samples
    discount             : float
    concentration        : float
    time_elapsed_numeric : float
    N : int
    S : int
    J : int

    def generate_conditional_posterior_predictive_zetas(self) -> npt.NDArray[np.float64]:
        zetas = np.swapaxes(np.array([
            zeta[delta]
            for zeta, delta
            in zip(self.samples.zeta, self.samples.delta)
            ]), 0, 1)
        return zetas
    
    def generate_conditional_posterior_predictive_gammas(self) -> npt.NDArray[np.float64]:
        zetas = self.generate_conditional_posterior_predictive_zetas()
        return gamma(shape = zetas)
    
    def generate_posterior_predictive_zetas(self, n_per_sample : int = 10) -> npt.NDArray[np.float64]:
        zetas = []
        nSamp = self.samples.chi.shape[0]     # number of MCMC samples
        probs = stickbreak(self.samples.chi)
        Sprob = np.cumsum(probs, axis = -1)
        unis  = uniform(size = (nSamp, n_per_sample))
        for s in range(nSamp):
            delta = np.searchsorted(Sprob[s], unis[s])
            zetas.append(self.samples.zeta[s][delta])
        zetas = np.vstack(zetas)
        return(zetas)
    
    def generate_posterior_predictive_gammas(self, n_per_sample : int = 10) -> npt.NDArray[np.float64]:
        zetas = self.generate_posterior_predictive_zetas(n_per_sample)
        return gamma(shape = zetas)

    def generate_posterior_predictive_hypercube(self, n_per_sample : int = 10) -> npt.NDArray[np.float64]:
        gammas = self.generate_posterior_predictive_gammas(n_per_sample)
        return euclidean_to_hypercube(gammas)

    def load_data(self, out):
        self.data    = Data.from_dict(out['data'])
        self.samples = Samples.from_dict(out['samples'])
        self.varparm = VariationalParameters.from_dict(out['varparm'])
        self.priors  = out['prior']
        self.time_elapsed_numeric = out['time']
        return

    def __init__(self, path):
        self.load_data(path)
        return
    
if __name__ == '__main__':
    # raw = pd.read_csv('./datasets/ivt_updated_nov_mar.csv')
    import pandas as pd
    raw = pd.read_csv('./datasets/ivt_nov_mar.csv').values
    data = Data.from_raw(
        raw, 
        x1ht_cols = np.arange(raw.shape[1]), 
        dcls = True, 
        xhquant = 0.95,
        )
    model = Chain(data, p = 10, gibbs_samples = 1000,)
    model.sample(5000, verbose = True)
    model.write_to_disk('./test/results.pkl')
    res = Result('./test/results.pkl')
    cond_zetas  = res.generate_conditional_posterior_predictive_zetas()
    cond_gammas = res.generate_conditional_posterior_predictive_gammas()
    zetas       = res.generate_posterior_predictive_zetas()
    gammas      = res.generate_posterior_predictive_gammas()
    raise

# EOF