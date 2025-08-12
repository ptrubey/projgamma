import numpy as np
import numpy.typing as npt
np.seterr(divide='raise', over = 'raise', under = 'ignore', invalid = 'raise')
from numpy.random import gamma, uniform, beta, normal
from collections import namedtuple
from typing import Self, NamedTuple

from .priors import GEMPrior, GammaPrior
from .samplers import StickBreakingSampler, SamplesBase, stickbreak,            \
    py_sample_chi_bgsb, py_sample_cluster_bgsb
from .density_gamma import logp_gamma_gamma_logshape_summary,                   \
    logp_resgamma_gamma_logshape_summary, logd_projresgamma_my_mt
from .data import Data, Projection, euclidean_to_hypercube

class Prior(NamedTuple):
    alpha : GammaPrior
    beta  : GammaPrior
    chi   : GEMPrior

class Samples(SamplesBase):
    zeta  : npt.NDArray[np.float64] 
    alpha : npt.NDArray[np.float64] 
    beta  : npt.NDArray[np.float64]
    chi   : npt.NDArray[np.float64]
    r     : npt.NDArray[np.float64]
    delta : npt.NDArray[np.int32]

    def to_dict(self, nBurn, nThin) -> dict:
        out = {
            'zeta'  : self.zeta[nBurn :: nThin, 0],
            'alpha' : self.alpha[nBurn :: nThin, 0],
            'beta'  : self.beta[nBurn :: nThin, 0],
            'chi'   : self.chi[nBurn :: nThin, 0],
            'delta' : self.delta[nBurn :: nThin, 0],
            'r'     : self.r[nBurn :: nThin, 0],
            }
        return out
    
    @classmethod
    def from_meta(
            cls, 
            nSamp  : int, 
            nDat   : int, 
            nCol   : int, 
            nClust : int,
            nTemp  : int
            ) -> Self:
        params = {
            'zeta'  : np.empty((nSamp + 1, nTemp, nClust, nCol)),
            'alpha' : np.empty((nSamp + 1, nTemp, nCol)),
            'beta'  : np.empty((nSamp + 1, nTemp, nCol)),
            'chi'   : np.empty((nSamp + 1, nTemp, nClust - 1)),
            'delta' : np.empty((nSamp + 1, nTemp, nDat), dtype = int),
            'r'     : np.empty((nSamp + 1, nTemp, nDat)),
            }
        return cls.from_dict(params)
    
    def __init__(
            self, 
            zeta  : npt.NDArray[np.float64], 
            alpha : npt.NDArray[np.float64], 
            chi   : npt.NDArray[np.float64], 
            beta  : npt.NDArray[np.float64], 
            delta : npt.NDArray[np.int32], 
            r     : npt.NDArray[np.float64],
            ):
        self.zeta  = zeta
        self.alpha = alpha
        self.beta  = beta
        self.chi   = chi
        self.delta = delta
        self.r     = r
        return

class Chain(StickBreakingSampler, Projection):
    samples       : Samples
    data          : Data
    concentration : float
    discount      : float
    priors        : Prior

    @property
    def curr_zeta(self) -> npt.NDArray[np.float64]:
        return self.samples.zeta[self.curr_iter]
    @property
    def curr_alpha(self) -> npt.NDArray[np.float64]:
        return self.samples.alpha[self.curr_iter]
    @property
    def curr_beta(self) -> npt.NDArray[np.float64]:
        return self.samples.beta[self.curr_iter]
    @property
    def curr_r(self) -> npt.NDArray[np.float64]:
        return self.samples.r[self.curr_iter]
    @property
    def curr_delta(self) -> npt.NDArray[np.int32]:
        return self.samples.delta[self.curr_iter]
    @property
    def curr_chi(self) -> npt.NDArray[np.float64]:
        return self.samples.chi[self.curr_iter]
   
    def sample_alpha(
            self,
            delta : npt.NDArray[np.int32], 
            zeta  : npt.NDArray[np.float64], 
            alpha : npt.NDArray[np.float64],
            ) -> npt.NDArray[np.float64]:
        """ 
        Samples hierarchical shape parameter for zeta.
            assumes rate parameter (beta) integrated out.
        """
        active_zeta = zeta[np.unique(delta)]
        n = np.array([active_zeta.shape[0]])
        zs = active_zeta.sum(axis = 0)
        lzs = np.log(active_zeta).sum(axis = 0)
        
        la_curr = np.log(alpha)
        la_cand = np.log(alpha) + normal(scale = 0.15, size = alpha.shape)

        logp = lambda logalpha: logp_gammagamma_logshape_summary(
            logalpha, lzs, zs, n, *self.priors.alpha, *self.priors.beta,
            )
        lfc_curr = logp(la_curr)
        lfc_cand = logp(la_cand)
        # lfc_curr = log_fc_log_alpha_k_summary(
        #     la_curr, np.array(n), zs, lzs, self.priors.alpha, self.priors.beta,
        #     )
        # lfc_cand = log_fc_log_alpha_k_summary(
        #     la_cand, np.array(n), zs, lzs, self.priors.alpha, self.priors.beta,
        #     )
        accept = np.log(uniform(size = alpha.shape)) < (lfc_cand - lfc_curr)
        la_curr[accept] = la_cand[accept]
        return np.exp(la_curr)

    def sample_beta(
            self, 
            delta : npt.NDArray[np.int32],
            zeta  : npt.NDArray[np.float64], 
            alpha : npt.NDArray[np.float64],
            ) -> npt.NDArray[np.float64]:
        """
        Samples hierarchical rate parameter for zeta.
        """
        active_zeta = zeta[np.unique(delta)]
        n = active_zeta.shape[0]
        zs = active_zeta.sum(axis = 0)
        As = n * alpha + self.priors.beta.a
        Bs = zs + self.priors.beta.b
        return gamma(shape = As, scale = 1 / Bs)

    def sample_zeta(
            self, 
            delta : npt.NDArray[np.int32],
            r     : npt.NDArray[np.float64], 
            zeta  : npt.NDArray[np.float64], 
            alpha : npt.NDArray[np.float64], 
            beta  : npt.NDArray[np.float64],
            ) -> npt.NDArray[np.float64]:
        """ 
        Samples shape parameters for projected (restricted) gamma.
        """
        dmat = delta[:,None] == np.arange(self.max_clust_count)
        Y    = r[:,None] * self.data.Yp
        n    = dmat.sum(axis = 0)
        Ys   = (Y.T @ dmat).T
        lYs  = (np.log(Y).T @ dmat).T
        lz_curr = np.log(zeta)
        lz_cand = lz_curr + normal(scale = 0.2, size = lz_curr.shape)

        logp = lambda logzeta: logp_resgamma_gamma_logshape_summary(
            logzeta, lYs, Ys, n[:,None], alpha, beta,
            )
        lfc_curr = logp(lz_curr)
        lfc_cand = logp(lz_cand)
        # lfc_curr = log_fc_log_alpha_1_summary(lz_curr, n[:,None], Ys, lYs, GammaPrior(alpha, beta))
        # lfc_cand = log_fc_log_alpha_1_summary(lz_cand, n[:,None], Ys, lYs, GammaPrior(alpha, beta))
        accept = np.log(uniform(size = alpha.shape)) < (lfc_cand - lfc_curr)
        lz_curr[accept] = lz_cand[accept]
        z_new = np.exp(lz_curr)
        
        resamp = (n == 0)
        z_new[resamp] = gamma(
            shape = alpha, scale = 1 / beta, size = z_new[resamp].shape,
            )
        return z_new

    def sample_r(
            self, 
            delta : npt.NDArray[np.int32], 
            zeta  : npt.NDArray[np.float64],
            ) -> npt.NDArray[np.float64]:
        """
        samples latent radii to recover independent gammas
        representation of projected (restricted) gamma.
        """
        As = np.einsum('il->i', zeta[delta])
        Bs = np.einsum('il->i', self.data.Yp)
        return gamma(shape = As, scale = 1 / Bs)
    
    def sample_chi(
            self, 
            delta : npt.NDArray[np.int32], 
            ) -> npt.NDArray[np.float64]:
        """
        Samples stick-breaking unnormalized cluster weights.
        """
        chi = py_sample_chi_bgsb(
            delta,
            disc = self.priors.chi.discount,
            conc = self.priors.chi.concentration,
            trunc = self.max_clust_count,
            )
        return chi
    
    def sample_delta(
            self,
            chi  : npt.NDArray[np.float64],
            zeta : npt.NDArray[np.float64],
            ) -> npt.NDArray[np.int32]:
        """
        Samples latent cluster identifiers
        """
        llk = logd_projresgamma_my_mt(self.data.Yp, zeta)
        # return py_sample_cluster_bgsb(chi, ll)
        # llk = np.zeros((self.nDat, self.max_clust_count))
        
        # logd_projgamma_my_mt_inplace_unstable(
        #     llk, self.data.Yp, zeta, np.ones(zeta.shape),
        #     )
        return py_sample_cluster_bgsb(chi, llk)

    def initialize_sampler(self, ns) -> None:
        self.curr_iter = 0
        self.samples = Samples.from_meta(ns, self.nDat, self.nCol, self.max_clust_count)
        self.samples.alpha[0] = 1.
        self.samples.beta[0] = 1.
        self.samples.zeta[0] = gamma(
            shape = 2., scale = 2., 
            size = (self.max_clust_count, self.nCol)
            )
        self.samples.chi[0] = beta(
            1 - self.priors.chi.discount, 
            self.priors.chi.concentration + self.priors.chi.discount * np.arange(1, self.max_clust_count),
            )
        self.samples.delta[0] = self.sample_delta(self.curr_chi, self.curr_zeta)
        self.samples.r[0] = self.sample_r(self.curr_delta, self.curr_zeta)
        return

    def iter_sample(self) -> None:
        delta = self.curr_delta
        alpha = self.curr_alpha
        beta  = self.curr_beta
        zeta  = self.curr_zeta

        self.curr_iter += 1

        self.samples.chi[self.curr_iter]   = self.sample_chi(delta)
        self.samples.delta[self.curr_iter] = self.sample_delta(self.curr_chi, zeta)
        self.samples.r[self.curr_iter]     = self.sample_r(self.curr_delta, zeta)
        self.samples.zeta[self.curr_iter]  = self.sample_zeta(
            self.curr_delta, self.curr_r, zeta, alpha, beta,
            )
        self.samples.alpha[self.curr_iter] = self.sample_alpha(
            self.curr_delta, self.curr_zeta, alpha,
            )
        self.samples.beta[self.curr_iter]  = self.sample_beta(
            self.curr_delta, self.curr_zeta, self.curr_alpha,
            )
        return
    
    def to_dict(self, nBurn = 0, nThin = 1) -> dict:
        out = {
            'samples' : self.samples.to_dict(nBurn, nThin),
            'data'    : self.data.to_dict(),
            'prior'   : self.priors,
            }
        return out 

    def __init__(
            self,
            data        : Data,
            prior_alpha : GammaPrior = GammaPrior(0.5, 0.5),
            prior_beta  : GammaPrior = GammaPrior(2., 2.),
            prior_chi   : GEMPrior   = GEMPrior(0.1, 0.1),
            p           : int        = 10,   # projection norm power
            max_clust   : int        = 200,  # stick-breaking truncation point
            temps       : int        = 1,    # number of tempering temps
            ladder      : float      = 1.2,  # tempering ladder base
            **kwargs
            ):
        """
        data        : Data,
        prior_alpha : GammaPrior
        prior_beta  : GammaPrior
        prior_chi   : GEMPrior
        p           : projection norm power (L_p norm)
        max_clust   : stick-breaking truncation point
        temps       : number of tempering temps
        ladder      : tempering ladder base
        ----
        Pitman-Yor Mixture of Projected (Restricted) Gammas, Gamma Prior.
        Tempering Temperatures are calculated as 1 / ladder^(0:temps)
        """
        self.data = data
        self.max_clust_count = max_clust
        self.p = p
        self.nCol = self.data.nCol
        self.nDat = self.data.nDat
        self.priors = Prior(
            GammaPrior(*[np.ndarray([x]) for x in prior_alpha]),
            GammaPrior(*[np.ndarray([x]) for x in prior_beta]),
            GEMPrior(*[np.ndarray([x]) for x in prior_chi])
            )
        self.set_projection()
        self.T = 
        return

class Result(object):
    def generate_conditional_posterior_predictive_zetas(self) -> npt.NDArray[np.float64]:
        """ rho | zeta, delta + W ~ Gamma(rho | zeta[delta] + W) """
        zetas = np.swapaxes(np.array([
            zeta[delta]
            for delta, zeta 
            in zip(self.samples.delta, self.samples.zeta)
            ]),0,1) # (n,s,d)
        return zetas
    
    def generate_conditional_posterior_predictive_gammas(self) -> npt.NDArray[np.float64]:
        """ rho | zeta, delta + W ~ Gamma(rho | zeta[delta] + W) """
        zetas = np.swapaxes(np.array([
            zeta[delta]
            for delta, zeta 
            in zip(self.samples.delta, self.samples.zeta)
            ]),0,1) # (n,s,d)
        return gamma(shape = zetas)

    def generate_posterior_predictive_zetas(
            self, n_per_sample = 1, *args, **kwargs
            ) -> npt.NDArray[np.float64]:
        zetas = []
        cumprob = stickbreak(self.samples.chi).cumsum(axis = -1)
        for s in range(self.nSamp):
            delta = np.searchsorted(cumprob[s], uniform(size = n_per_sample))
            zetas.append(self.samples.zeta[s][delta])
        return np.vstack(zetas)

    def generate_posterior_predictive_gammas(
            self, n_per_sample = 1, m = 10, *args, **kwargs
            ) -> npt.NDArray[np.float64]:
        zetas = self.generate_posterior_predictive_zetas(n_per_sample, m, *args, **kwargs)
        return gamma(shape = zetas)

    def generate_posterior_predictive_hypercube(
            self, n_per_sample = 1, m = 10, *args, **kwargs
            ) -> npt.NDArray[np.float64]:
        gammas = self.generate_posterior_predictive_gammas(n_per_sample, m, *args, **kwargs)
        return euclidean_to_hypercube(gammas)

    def load_data(self, out : dict) -> None:        
        self.samples = Samples.from_dict(out['samples'])
        self.data    = Data.from_dict(out['data'])
        self.priors  = out['prior']
        self.nSamp = self.samples.delta.shape[0]
        self.nDat  = self.samples.delta.shape[1]
        self.nCol  = self.samples.alpha.shape[1]
        return

    def __init__(self, out):
        self.load_data(out)
        return

if __name__ == '__main__':
    pass


# EOF
