import numpy as np
import numpy.typing as npt
np.seterr(divide='raise', over = 'raise', under = 'ignore', invalid = 'raise')
from numpy.random import gamma, uniform, beta, normal
from collections import namedtuple
from typing import Self

from samplers import GEMPrior, GammaPrior, StickBreakingSampler,              \
    py_sample_chi_bgsb, py_sample_cluster_bgsb
from projgamma import log_fc_log_alpha_1_summary, log_fc_log_alpha_k_summary, \
    pt_logd_projgamma_my_mt
from data import Data, Projection, euclidean_to_hypercube
Prior = namedtuple('Prior', 'alpha beta chi')

class Samples(object):
    zeta  : npt.NDArray[np.float64] = None
    alpha : npt.NDArray[np.float64] = None
    beta  : npt.NDArray[np.float64] = None
    chi   : npt.NDArray[np.float64] = None
    r     : npt.NDArray[np.float64] = None
    delta : npt.NDArray[np.int32]   = None

    def to_dict(self, nBurn, nThin) -> dict:
        out = {
            'zeta'  : self.zeta[nBurn :: nThin],
            'alpha' : self.alpha[nBurn :: nThin],
            'beta'  : self.beta[nBurn :: nThin],
            'chi'   : self.chi[nBurn :: nThin],
            'delta' : self.delta[nBurn :: nThin],
            'r'     : self.r[nBurn :: nThin],
            }
        return out
    
    @classmethod
    def from_dict(cls, out) -> Self:
        return cls(**out)

    @classmethod
    def from_parameters(
            cls, 
            nSamp  : int, 
            nDat   : int, 
            nCol   : int, 
            nClust : int,
            ) -> Self:
        params = {
            'zeta'  : np.empty((nSamp + 1, nClust, nCol)),
            'alpha' : np.empty((nSamp + 1, nCol)),
            'beta'  : np.empty((nSamp + 1, nCol)),
            'chi'   : np.empty((nSamp + 1, nClust - 1)),
            'delta' : np.empty((nSamp + 1, nDat), dtype = int),
            'r'     : np.empty((nSamp + 1, nDat)),
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
    priors        : Prior[GammaPrior, GammaPrior, GEMPrior]

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
        n = active_zeta.shape[0]
        zs = active_zeta.sum(axis = 0)
        lzs = np.log(active_zeta).sum(axis = 0)
        la_curr = np.log(alpha)
        la_cand = np.log(alpha) + normal(scale = 0.15, size = alpha.shape)
        lfc_curr = log_fc_log_alpha_k_summary(
            la_curr, np.array(n), zs, lzs, self.priors.alpha, self.priors.beta,
            )
        lfc_cand = log_fc_log_alpha_k_summary(
            la_cand, np.array(n), zs, lzs, self.priors.alpha, self.priors.beta,
            )
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
        Ys  = (Y.T @ dmat).T
        lYs = (np.log(Y).T @ dmat).T
        lz_curr = np.log(zeta)
        lz_cand = lz_curr + normal(scale = 0.2, size = lz_curr.shape)

        lfc_curr = log_fc_log_alpha_1_summary(lz_curr, n[:,None], Ys, lYs, alpha, beta)
        lfc_cand = log_fc_log_alpha_1_summary(lz_cand, n[:,None], Ys, lYs, alpha, beta)

        accept = np.log(uniform(size = alpha.shape)) < lfc_cand - lfc_curr
        accept = uniform(size = alpha.shape) < np.exp(lfc_cand - lfc_curr)
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
        return py_sample_chi_bgsb(
            delta,
            disc = self.priors.chi.discount,
            conc = self.priors.chi.concentration,
            trunc = self.max_clust_count,
            )
    
    def sample_delta(
            self,
            chi  : npt.NDArray[np.float64],
            zeta : npt.NDArray[np.float64],
            ) -> npt.NDArray[np.int32]:
        """
        Samples latent cluster identifiers
        """
        ll = pt_logd_projgamma_my_mt(self.data.Yp, zeta)
        return py_sample_cluster_bgsb(chi, ll)

    def initialize_sampler(self, ns) -> None:
        self.curr_iter = 0
        self.samples = Samples.from_parameters(ns, self.nDat, self.nCol)
        self.samples.alpha[0] = 1.
        self.samples.beta[0] = 1.
        self.samples.zeta[0] = gamma(
            shape = 2., scale = 2., 
            size = (self.max_clust_count, self.nCol)
            )
        self.samples.chi[0] = beta(
            1 - self.discount, 
            self.concentration + self.discount * np.arange(1, self.max_clust_count),
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
            'conc'    : self.concentration,
            'disc'    : self.discount
            }
        return out 

    def __init__(
            self,
            data        : Data,
            prior_alpha : GammaPrior = GammaPrior(0.5, 0.5),
            prior_beta  : GammaPrior = GammaPrior(2., 2.),
            prior_chi   : GEMPrior   = GEMPrior(0.1, 0.1),
            p           : int        = 10,
            max_clust   : int        = 200,
            **kwargs
            ):
        self.data = data
        self.max_clust_count = max_clust
        self.p = p
        self.nCol = self.data.nCol
        self.nDat = self.data.nDat
        self.priors = Prior(
            GammaPrior(*prior_alpha), 
            GammaPrior(*prior_beta), 
            GEMPrior(*prior_chi),
            )
        self.set_projection()
        return

class Result(object):
    def generate_conditional_posterior_predictive_zetas(self):
        """ rho | zeta, delta + W ~ Gamma(rho | zeta[delta] + W) """
        zetas = np.swapaxes(np.array([
            zeta[delta]
            for delta, zeta 
            in zip(self.samples.delta, self.samples.zeta)
            ]),0,1) # (n,s,d)
        return zetas
    
    def generate_conditional_posterior_predictive_gammas(self):
        """ rho | zeta, delta + W ~ Gamma(rho | zeta[delta] + W) """
        zetas = np.swapaxes(np.array([
            zeta[delta]
            for delta, zeta 
            in zip(self.samples.delta, self.samples.zeta)
            ]),0,1) # (n,s,d)
        return gamma(shape = zetas)

    def generate_posterior_predictive_zetas(self, n_per_sample = 1, m = 10, *args, **kwargs):
        raise NotImplementedError('Fix this!')
        zetas = []
        for s in range(self.nSamp):
            dmax = self.samples.delta[s].max()
            njs = np.bincount(self.samples.delta[s], minlength = int(dmax + 1 + m))
            ljs = (
                + njs - (njs > 0) * self.discount 
                + (njs == 0) * (
                    self.concentration 
                    + (njs > 0).sum() * self.discount
                    ) / m
                )
            new_zetas = gamma(
                shape = self.samples.alpha[s],
                scale = 1. / self.samples.beta[s],
                size = (m, self.nCol),
                )
            prob = ljs / ljs.sum()
            deltas = generate_indices(prob, n_per_sample)
            zetas.append(np.vstack((self.samples.zeta[s], new_zetas))[deltas])
        return np.vstack(zetas)

    def generate_posterior_predictive_gammas(self, n_per_sample = 1, m = 10, *args, **kwargs):
        zetas = self.generate_posterior_predictive_zetas(n_per_sample, m, *args, **kwargs)
        return gamma(shape = zetas)

    def generate_posterior_predictive_hypercube(self, n_per_sample = 1, m = 10, *args, **kwargs):
        gammas = self.generate_posterior_predictive_gammas(n_per_sample, m, *args, **kwargs)
        return euclidean_to_hypercube(gammas)

    def load_data(self, out : dict) -> None:        
        self.samples = Samples.from_dict(out['samples'])
        self.data    = Data.from_dict(out['data'])
        self.priors  = out['prior']
        self.concentration = out['conc']
        self.discount      = out['disc']
        self.nSamp = self.samples.delta.shape[0]
        self.nDat  = self.samples.delta.shape[1]
        self.nCol  = self.samples.alpha.shape[1]
        return

    def __init__(self, path):
        self.load_data(path)
        return

if __name__ == '__main__':
    pass

    from data import Data_From_Raw
    from projgamma import GammaPrior
    from pandas import read_csv
    import os

    raw = read_csv('./datasets/ivt_nov_mar.csv')
    data = Data_From_Raw(raw, decluster = True, quantile = 0.95)
    model = Chain(data, p = 10)
    model.sample(10000, verbose = True)
    model.write_to_disk('./test/results.pkl', 5000, 2)
    res = Result('./test/results.pkl')

# EOF
