import numpy as np
import numpy.typing as npt
from typing import Self, NamedTuple
from numpy.random import choice, gamma, uniform, normal, lognormal
from collections import deque
from scipy.special import digamma

# Custom Modules
from .priors import GammaPrior, GEMPrior
from .samplers import StickBreakingSampler, VariationalBase, SamplesBase,       \
    stickbreak, py_sample_chi_bgsb, py_sample_cluster_bgsb
from .varbayes import Adam
from .data import euclidean_to_hypercube, Data, Projection
from .density_gamma import logd_projresgamma_my_mt

np.seterr(divide = 'raise', over = 'raise', under = 'ignore', invalid = 'raise')

class Prior(NamedTuple):
    alpha : GammaPrior
    beta  : GammaPrior
    chi   : GEMPrior

class Samples(SamplesBase):
    r     : deque[npt.NDArray[np.float64]] | npt.NDArray[np.float64] # radius (projected gamma)
    chi   : deque[npt.NDArray[np.float64]] | npt.NDArray[np.float64] # stick-breaking weights (unnormalized)
    delta : deque[npt.NDArray[np.int32]]   | npt.NDArray[np.int32]   # cluster identifiers
    beta  : deque[npt.NDArray[np.float64]] | npt.NDArray[np.float64] # rate hyperparameter
    alpha : deque[npt.NDArray[np.float64]] | npt.NDArray[np.float64] # shape hyperparameter (inferred)
    zeta  : deque[npt.NDArray[np.float64]] | npt.NDArray[np.float64] # shape parameter (inferred)
    
    def to_dict(self) -> dict:
        """
        Samples.to_dict():
            Creates a dictionary object of the current Samples state.
        """
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
    def from_meta(cls, nkeep : int, N : int, S : int, J : int) -> Self:
        """
        Samples normal initialization routine.
        args:
            nkeep: number of kept samples
            N: number of observations
            S: number of dimensions
            J: maximum number of extant clusters
        """
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

def grad_resgamgam_ln(
        theta   : npt.NDArray[np.float64],  # np.stack((mu, tau))    # (2, j, d)
        lYs     : npt.NDArray[np.float64],  # sum of log(Y)          # (j, d)
        n       : npt.NDArray[np.int32],    # number of observations # (j)
        a       : npt.NDArray[np.float64] | float,  # hierarchical shape     # (d) or float
        b       : npt.NDArray[np.float64] | float,  # hierarchical rate      # (d) or float
        ns      : int = 20,
        ) -> npt.NDArray[np.float64]:
    """
    grad_resgamgam_ln(...):
        gradient for distributional parameters of shape parameter (log-scale), 
            for gamma data with gamma prior on shape, where rate := 1.
    args:
        theta : distributional parameters for shape parameter (mu, tau)
        lYs   : log(Y) summed
        Ys    : Y summed
        n     : number of observations of Y
        a,b   : hierarchical shape parameters (for gamma prior on shape)
        ns    : number of samples per iteration
    output:
        dtheta: gradient of objective at theta... returning mean of ns samples.
    """
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

def grad_gamgam_ln(
        theta   : npt.NDArray[np.float64],  # np.stack((mu, tau))
        lYs     : npt.NDArray[np.float64],  # sum of log(Y)
        Ys      : npt.NDArray[np.float64],  # sum of Y
        n       : npt.NDArray[np.int32],    # number of observations
        a       : npt.NDArray[np.float64],  # hierarchical (shape) shape 
        b       : npt.NDArray[np.float64],  # hierarchical (shape) rate
        c       : npt.NDArray[np.float64],  # hierarchical (rate) shape
        d       : npt.NDArray[np.float64],  # hierarchical (rate) rate
        ns      : int = 10,
        ) -> npt.NDArray[np.float64]:
    """
    grad_gamgam_ln(...):
        gradient for distributional parameters of shape parameter (log-scale), 
            for gamma data with gamma priors on shape, rate, where where rate 
            parameter has been integrated out.
    args:
        theta : distributional parameters for shape parameter (mu, tau)
        lYs   : log(Y) summed
        Ys    : Y summed
        n     : number of observations of Y
        a,b   : hierarchical shape parameters (for gamma prior on shape)
        c,d   : hierarchical rate parameters (for gamma prior on rate)
        ns    : number of samples per iteration
    output:
        dtheta: gradient of objective at theta... returning mean of ns samples.
    """
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

class VariationalParameters(VariationalBase):
    logzeta : Adam
    logalpha : Adam

    def to_dict(self) -> dict:
        out = {
            'logzeta'   : self.logzeta.to_dict(),
            'logalpha' : self.logalpha.to_dict(),
            }
        return out
    
    @classmethod
    def from_meta(cls, S : int, J : int, **kwargs) -> Self:
        """
        Initialization of Variational Parameters from meta-information
        ---
        args:
            S : number of dimensions
            J : Max number of extant clusters
        """
        logzeta_mutau = np.zeros((2, J, S))
        logalpha_mutau = np.zeros((2, S))
        logzeta = Adam.from_meta(logzeta_mutau, **kwargs)
        logalpha = Adam.from_meta(logalpha_mutau, **kwargs)
        return cls(logzeta = logzeta, logalpha = logalpha)

    @classmethod
    def from_dict(cls, out : dict) -> Self:
        """
        Initialization of Variational Parameters from dictionary
        args:
            out (output of VariationalParameters.to_dict())
        """
        logzeta = Adam.from_dict(out['logzeta'])
        logalpha = Adam.from_dict(out['logalpha'])
        return cls(logzeta = logzeta, logalpha = logalpha)

    def __init__(
            self,
            logzeta : Adam,
            logalpha : Adam,
            ):
        self.logzeta = logzeta
        self.logalpha = logalpha
        return
    pass

class Chain(StickBreakingSampler, Projection):
    samples       : Samples
    varparm       : VariationalParameters
    priors        : Prior
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
        
        func = lambda theta: - grad_resgamgam_ln(
            theta, lYs, n, alpha, beta, self.var_samp,
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
        """
        Variational approximation for hierarchical shape parameter for zeta.
        Assume beta has been integrated out.
        """
        active = np.where(np.bincount(delta, minlength = self.J) > 0)[0]
        n = np.array((active.shape[0],))
        lZs = np.log(zeta)[active].sum(axis = 0)
        Zs  = zeta[active].sum(axis = 0)

        func = lambda theta: - grad_gamgam_ln(
            theta, 
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
        """
        Samples hierarchical rate parameter for zeta.
        """
        active_zeta = zeta[np.unique(delta)]
        n = active_zeta.shape[0]
        zs = active_zeta.sum(axis = 0)
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
        # r[r < 1e-4] = 1e-4               # lower-bounding radius
        self.samples.r.append(r)
        return
    
    def update_chi(self, delta : npt.NDArray[np.int32]) -> None:
        chi = py_sample_chi_bgsb(
            delta, 
            self.priors.chi.discount, 
            self.priors.chi.concentration, 
            self.J,
            )
        self.samples.chi.append(chi)
        return
    
    def update_delta(
            self, 
            zeta : npt.NDArray[np.float64], 
            chi  : npt.NDArray[np.float64],
            ) -> None:
        llk = logd_projresgamma_my_mt(y = self.data.Yp, shape = zeta)
        delta = py_sample_cluster_bgsb(chi = chi, log_likelihood = llk)
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
        self.samples = Samples.from_meta(self.gibbs_samp, self.N, self.S, self.J)
        self.varparm = VariationalParameters.from_meta(
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

    def load_data(self, out : dict) -> None:
        self.data    = Data.from_dict(out['data'])
        self.samples = Samples.from_dict(out['samples'])
        self.varparm = VariationalParameters.from_dict(out['varparm'])
        self.priors  = out['prior']
        self.time_elapsed_numeric = out['time']
        return

    def __init__(self, out : dict):
        self.load_data(out)
        return
    
if __name__ == '__main__':
    pass

# EOF