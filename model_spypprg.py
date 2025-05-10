import numpy as np
import numpy.typing as npt
np.seterr(divide='raise', over = 'raise', under = 'ignore', invalid = 'raise')

from numpy.random import choice, gamma, uniform
from itertools import repeat
from collections import namedtuple
from scipy.special import gammaln
from typing import Self

from samplers import StickBreakingSampler, py_sample_chi_bgsb, py_sample_cluster_bgsb
from projgamma import sample_alpha_1_mh, sample_alpha_k_mh
from data import Data, euclidean_to_hypercube
from projgamma import GammaPrior, logd_prodgamma_my_mt, logd_prodgamma_paired,  \
    logd_prodgamma_my_st, logd_gamma

def update_zeta_j_wrapper(args):
    # parse arguments
    curr_zeta_j, n_j, Y_js, lY_js, alpha, beta = args
    prop_zeta_j = np.empty(curr_zeta_j.shape)
    for i in range(curr_zeta_j.shape[0]):
        prop_zeta_j[i] = sample_alpha_1_mh_summary(
            curr_zeta_j[i], n_j, Y_js[i], lY_js[i], alpha[i], beta[i],
            )
    return prop_zeta_j

def sample_gamma_shape_wrapper(args):
    return sample_alpha_k_mh_summary(*args)

Prior = namedtuple('Prior', 'alpha beta')

class Samples(object):
    zeta  = None
    alpha = None
    beta  = None
    delta = None
    r     = None

    def to_dict(self, nBurn, nThin) -> dict:
        out = {
            'zeta'  : self.samples.zeta[nBurn :: nThin],
            'alpha' : self.samples.alpha[nBurn :: nThin],
            'beta'  : self.samples.beta[nBurn :: nThin],
            'delta' : self.samples.delta[nBurn :: nThin],
            'r'     : self.samples.r[nBurn :: nThin],
            }
        return out
    
    @classmethod
    def from_dict(cls, out) -> Self:
        return cls(**out)

    @classmethod
    def from_parameters(cls, nSamp, nDat, nCol, nClust) -> Self:
        params = {
            'zeta'  : np.empty((nSamp + 1, nClust, nCol)),
            'alpha' : np.empty((nSamp + 1, nCol)),
            'beta'  : np.empty((nSamp + 1, nCol)),
            'delta' : np.empty((nSamp + 1, nDat), dtype = int),
            'r'     : np.empty((nSamp + 1, nDat)),
            }
        return cls.from_dict(params)
    
    def __init__(self, zeta, alpha, beta, delta, r):
        self.zeta  = zeta
        self.alpha = alpha
        self.beta  = beta
        self.delta = delta
        self.r     = r
        return

class Chain(StickBreakingSampler):
    concentration = None
    discount      = None

    @property
    def curr_zeta(self):
        return self.samples.zeta[self.curr_iter]
    @property
    def curr_alpha(self):
        return self.samples.alpha[self.curr_iter]
    @property
    def curr_beta(self):
        return self.samples.beta[self.curr_iter]
    @property
    def curr_r(self):
        return self.samples.r[self.curr_iter]
    @property
    def curr_delta(self):
        return self.samples.delta[self.curr_iter]
    
    def clean_delta_zeta(self, delta, zeta):
        """
        delta : cluster indicator vector (n)
        zeta  : cluster parameter matrix (J* x d)
        sigma : cluster parameter matrix (J* x d)
        """
        # reindex those clusters
        keep, delta[:] = np.unique(delta, return_inverse = True)
        # return new indices, cluster parameters associated with populated clusters
        return delta, zeta[keep]

    def sample_zeta_new(self, alpha, beta, m):
        return gamma(shape = alpha, scale = 1/beta, size = (m, self.nCol))
    
    def sample_alpha(self, zeta, curr_alpha):
        n    = zeta.shape[0]
        zs   = zeta.sum(axis = 0)
        lzs  = np.log(zeta).sum(axis = 0)
        args = zip(
            curr_alpha, repeat(n), zs, lzs,
            repeat(self.priors.alpha.a), repeat(self.priors.alpha.b),
            repeat(self.priors.beta.a), repeat(self.priors.beta.b),
            )
        res = map(sample_gamma_shape_wrapper, args)
        return np.array(list(res))

    def sample_beta(self, zeta, alpha):
        n = zeta.shape[0]
        zs = zeta.sum(axis = 0)
        As = n * alpha + self.priors.beta.a
        Bs = zs + self.priors.beta.b
        return gamma(shape = As, scale = 1 / Bs)

    def sample_r(self, delta, zeta):
        As = np.einsum('il->i', zeta[delta])
        Bs = np.einsum('il->i', self.data.Yp)
        return gamma(shape = As, scale = 1 / Bs)
    
    def sample_zeta(self, curr_zeta, r, delta, alpha, beta):
        dmat = delta[:,None] == np.arange(delta.max() + 1)
        Y    = r[:,None] * self.data.Yp
        n    = dmat.sum(axis = 0)
        Ysv  = (Y.T @ dmat).T
        lYsv = (np.log(Y).T @ dmat).T
        args = zip(
            curr_zeta, n, Ysv, lYsv, 
            repeat(alpha), repeat(beta),
            )
        res = map(update_zeta_j_wrapper, args)
        return np.array(list(res))
    
    def initialize_sampler(self, ns):
        self.samples = Samples(ns, self.nDat, self.nCol)
        self.samples.alpha[0] = 1.
        self.samples.beta[0] = 1.
        self.samples.zeta[0] = gamma(
            shape = 2., scale = 2., 
            size = (self.max_clust_count - 30, self.nCol)
            )
        self.samples.delta[0] = choice(self.max_clust_count - 30, size = self.nDat)
        self.samples.delta[0][-1] = np.arange(self.max_clust_count - 30)[-1] # avoids an error in initialization
        self.samples.r[0] = self.sample_r(self.samples.delta[0], self.samples.zeta[0])
        self.curr_iter = 0
        self.sigma_ph1 = np.ones((self.max_clust_count, self.nCol))
        self.sigma_ph2 = np.ones((self.nDat, self.nCol))
        return
    
    def record_log_density(self):
        lpl = 0.
        lpp = 0.
        Y = self.curr_r[:,None] * self.data.Yp
        lpl += logd_prodgamma_paired(
            Y,
            self.curr_zeta[self.curr_delta],
            self.sigma_ph2,
            ).sum()
        lpl += logd_prodgamma_my_st(self.curr_zeta, self.curr_alpha, self.curr_beta).sum()
        lpp += logd_gamma(self.curr_alpha, *self.priors.alpha).sum()
        lpp += logd_gamma(self.curr_beta, *self.priors.beta).sum()
        self.samples.ld[self.curr_iter] = lpl + lpp
        return

    def iter_sample(self):
        # current cluster assignments; number of new candidate clusters
        delta = self.curr_delta.copy();  m = self.max_clust_count - (delta.max() + 1)
        alpha = self.curr_alpha
        beta  = self.curr_beta
        zeta  = np.vstack((self.curr_zeta, self.sample_zeta_new(alpha, beta, m)))
        r     = self.curr_r

        self.curr_iter += 1
        # Log-density for product of Gammas
        log_likelihood = logd_prodgamma_my_mt(r[:,None] * self.data.Yp, zeta, self.sigma_ph1)
        # pre-generate uniforms to inverse-cdf sample cluster indices
        unifs   = uniform(size = self.nDat)
        # Sample new cluster membership indicators 
        delta = pityor_cluster_sampler(
            delta, log_likelihood, unifs, self.concentration, self.discount,
            )
        # clean indices (clear out dropped clusters, unused candidate clusters, and re-index)
        delta, zeta = self.clean_delta_zeta(delta, zeta)
        self.samples.delta[self.curr_iter] = delta
        self.samples.r[self.curr_iter]     = self.sample_r(self.curr_delta, zeta)
        self.samples.zeta[self.curr_iter]  = self.sample_zeta(
                zeta, self.curr_r, self.curr_delta, alpha, beta,
                )
        self.samples.alpha[self.curr_iter] = self.sample_alpha(self.curr_zeta, alpha)
        self.samples.beta[self.curr_iter]  = self.sample_beta(self.curr_zeta, self.curr_alpha)

        self.record_log_density()
        return
    
    def to_dict(self, nBurn = 0, nThin = 1) -> dict:
        out = {
            'samples' : self.samples.to_dict(nBurn, nThin),
            'data'    : self.data.to_dict(),
            'prior'   : self.priors,
            }
        return out 

    def write_to_disk(self, path, nBurn, nThin = 1):
        if type(path) is str:
            folder = os.path.split(path)[0]
            if not os.path.exists(folder):
                os.mkdir(folder)
            if os.path.exists(path):
                os.remove(path)
        
        zetas  = np.vstack([
            np.hstack((np.ones((zeta.shape[0], 1)) * i, zeta))
            for i, zeta in enumerate(self.samples.zeta[nBurn :: nThin])
            ])
        alphas = self.samples.alpha[nBurn :: nThin]
        betas  = self.samples.beta[nBurn :: nThin]
        deltas = self.samples.delta[nBurn :: nThin]
        rs     = self.samples.r[nBurn :: nThin]

        out = {
            'zetas'  : zetas,
            'alphas' : alphas,
            'betas'  : betas,
            'rs'     : rs,
            'deltas' : deltas,
            'nCol'   : self.nCol,
            'nDat'   : self.nDat,
            'V'      : self.data.V,
            'logd'   : self.samples.ld,
            'time'   : self.time_elapsed_numeric,
            'conc'   : self.concentration,
            'disc'   : self.discount,
            }
        
        try:
            out['Y'] = self.data.Y
        except AttributeError:
            pass
        
        if type(path) is BytesIO:
            path.write(pickle.dumps(out))
        else:
            with open(path, 'wb') as file:
                pickle.dump(out, file)

        return

    def set_projection(self):
        self.data.Yp = (self.data.V.T / (self.data.V**self.p).sum(axis = 1)**(1/self.p)).T
        return

    def __init__(
            self,
            data,
            prior_alpha   = GammaPrior(0.5, 0.5),
            prior_beta    = GammaPrior(2., 2.),
            p             = 10,
            concentration = 0.2,
            discount      = 0.05,
            max_clust_count = 200,
            **kwargs
            ):
        self.concentration = concentration
        self.discount = discount
        self.data = data
        self.max_clust_count = max_clust_count
        self.p = p
        self.nCol = self.data.nCol
        self.nDat = self.data.nDat
        _prior_alpha = GammaPrior(*prior_alpha)
        _prior_beta = GammaPrior(*prior_beta)
        self.priors = Prior(_prior_alpha, _prior_beta)
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

    def generate_posterior_predictive_angular(self, n_per_sample = 1, m = 10, *args, **kwargs):
        hyp = self.generate_posterior_predictive_hypercube(n_per_sample, m, *args, **kwargs)
        return euclidean_to_angular(hyp)

    def write_posterior_predictive(self, path, n_per_sample = 1):
        thetas = pd.DataFrame(
                self.generate_posterior_predictive_angular(n_per_sample),
                columns = ['theta_{}'.format(i) for i in range(1, self.nCol)],
                )
        thetas.to_csv(path, index = False)
        return

    def load_data(self, path):
        if type(path) is BytesIO:
            out = pickle.loads(path.getvalue())
        else:
            with open(path, 'rb') as file:
                out = pickle.load(file)
        
        deltas = out['deltas']
        zetas  = out['zetas']
        alphas = out['alphas']
        betas  = out['betas']
        rs     = out['rs']
        conc   = out['conc']
        disc   = out['disc']

        self.concentration = conc
        self.discount      = disc
        self.nSamp = deltas.shape[0]
        self.nDat  = deltas.shape[1]
        self.nCol  = alphas.shape[1]

        self.data = Data_From_Sphere(out['V'])
        try:
            self.data.fill_outcome(out['Y'])
        except KeyError:
            pass

        self.samples       = Samples(self.nSamp, self.nDat, self.nCol)
        self.samples.delta = deltas
        self.samples.alpha = alphas
        self.samples.beta  = betas
        self.samples.zeta  = [
            zetas[np.where(zetas.T[0] == i)[0], 1:] for i in range(self.nSamp)
            ]
        self.samples.r     = rs
        self.samples.ld    = out['logd']
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
