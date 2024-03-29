from numpy.random import choice, gamma, beta, uniform, normal
from collections import namedtuple
from itertools import repeat, chain
import numpy as np
np.seterr(divide='raise', over = 'raise', under = 'ignore', invalid = 'raise')
import pandas as pd
import os
import pickle
from math import log
from scipy.special import gammaln, betaln

import cUtility as cu
from samplers import DirichletProcessSampler
from cProjgamma import sample_alpha_1_mh_summary, sample_alpha_k_mh_summary
from cUtility import diriproc_cluster_sampler
from data import euclidean_to_angular, euclidean_to_hypercube, euclidean_to_simplex, MixedData
from projgamma import GammaPrior
from model_mdpppg import BaseData, log_post_log_zeta_1

# from multiprocessing import Pool
# from energy import limit_cpu

def dprojresgamma_log_my_mt(aY, aAlpha):
    """
    projected restricted Gamma log-density for a single projection.
    ----
    aY     : array of Y     (n x d)
    aAlpha : array of alpha (J x d)
    ----
    return : array of ld    (n x J)
    """
    out = np.zeros((aY.shape[0], aAlpha.shape[0]))
    out -= np.einsum('jd->j', gammaln(aAlpha))[None,:]
    out += np.einsum('jd,nd->nj', aAlpha - 1, np.log(aY))
    out += gammaln(np.einsum('jd->j', aAlpha))[None,:]
    out -= np.einsum(
        'j,n->nj', np.einsum('jd->j', aAlpha), np.einsum('nd->n', aY),
        )
    return out

def dprgmultinom_log_mw_mt(aW, aAlpha):
    """
    projected restricted Gamma Multinomial distribution.
    ---
    aW     : array of W     (n x d)
    aAlpha : array of alpha (J x d)
    ---
    return : array of ld (n x j)
    """
    out = np.zeros((aW.shape[0], aAlpha.shape[0]))
    out += gammaln(np.einsum('jd->j', aAlpha))[None,:]
    out += gammaln(np.einsum('nd->n', aW))[:,None]
    out -= gammaln(
        + np.einsum('nd->n', aW)[:, None] 
        + np.einsum('jd->j', aAlpha)[None,:]
        )
    out += np.einsum('njd->nj', gammaln(aW[:,None,:] + aAlpha[None,:,:]))
    out -= np.einsum('jd->j', gammaln(aAlpha))[None,:]
    out -= np.einsum('nd->n', gammaln(aW + 1))[:,None]
    return out

def dprgmultinom_log_mw_mt_(aW, aAlpha):
    pass
    n = np.einsum('nd->n', aW)      # (n)
    a0 = np.einsum('jd->j', aAlpha) # (j)
    out = np.zeros((aW.shape[0], aAlpha.shape[0]))
    out += np.log(n)
    out += betaln(a0[None,:], n[:,None])
    #out -= (aW * betaln(aAlpha, aW)


def dprodgamma_log_my_mt(aY, aAlpha, aBeta):
    """
    Product of Gammas log-density for multiple Y, multiple theta (not paired)
    ----
    aY     : array of Y     (n x d)
    aAlpha : array of alpha (J x d)
    aBeta  : array of beta  (J x d)
    ----
    return : array of ld    (n x J)
    """
    out = np.zeros((aY.shape[0], aAlpha.shape[0]))
    out += np.einsum('jd,jd->j', aAlpha, np.log(aBeta)).reshape(1,-1) # beta^alpha
    out -= np.einsum('jd->j', gammaln(aAlpha)).reshape(1,-1)          # gamma(alpha)
    out += np.einsum('jd,nd->nj', aAlpha - 1, np.log(aY))             # y^(alpha - 1)
    out -= np.einsum('jd,nd->nj', aBeta, aY)                          # e^(-beta y)
    return out

def update_zeta_j_wrapper(args):
    # parse arguments
    curr_zeta_j, n_j, Y_js, lY_js, alpha, beta = args
    prop_zeta_j = np.empty(curr_zeta_j.shape)
    # iterate through zeta sampling
    for i in range(curr_zeta_j.shape[0]):
        prop_zeta_j[i] = sample_alpha_1_mh_summary(
            curr_zeta_j[i], n_j, Y_js[i], lY_js[i], alpha[i], beta[i]
            )
    return prop_zeta_j

def sample_gamma_shape_wrapper(args):
    return sample_alpha_k_mh_summary(*args)


Prior = namedtuple('Prior', 'eta alpha beta')

class Samples(object):
    pi    = None
    zeta  = None
    sigma = None
    alpha = None
    beta  = None
    xi    = None
    tau   = None
    delta = None
    r     = None
    eta   = None

    def __init__(self, nSamp, nDat, nCol, nCat, nCats):
        """
        nCol: number of 
        nCat: number of categorical columns
        nCats: number of categorical variables        
        """
        self.zeta  = [None] * (nSamp + 1)
        self.rho   = np.empty((nSamp + 1, nDat, nCat))
        self.alpha = np.empty((nSamp + 1, nCol + nCat))
        self.beta  = np.empty((nSamp + 1, nCol + nCat))
        self.delta = np.empty((nSamp + 1, nDat), dtype = int)
        self.r     = np.empty((nSamp + 1, nDat))
        self.eta   = np.empty(nSamp + 1)
        return

class Chain(DirichletProcessSampler):
    @property
    def curr_rho(self):
        return self.samples.rho[self.curr_iter]
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
    @property
    def curr_eta(self):
        return self.samples.eta[self.curr_iter]

    def sample_delta_i(self, curr_cluster_state, cand_cluster_state, eta, 
                                        log_likelihood_i, delta_i, p, scratch):
        scratch[:] = 0
        curr_cluster_state[delta_i] -= 1
        scratch += curr_cluster_state
        scratch += cand_cluster_state * (eta / (cand_cluster_state.sum() + 1e-9))
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            np.log(scratch, out = scratch)
        # scratch += np.log(curr_cluster_state + cand_cluster_state * eta / cand_cluster_state.sum())
        scratch += log_likelihood_i
        np.nan_to_num(scratch, False, -np.inf)
        scratch -= scratch.max()
        with np.errstate(under = 'ignore'):
            np.exp(scratch, out = scratch)
        np.cumsum(scratch, out = scratch)
        delta_i = np.searchsorted(scratch, p * scratch[-1])
        curr_cluster_state[delta_i] += 1
        cand_cluster_state[delta_i] = False
        return delta_i
    
    def clean_delta_zeta(self, delta, zeta):
        """
        delta : cluster indicator vector (n)
        zeta  : cluster parameter matrix (J* x d)
        """
        # which clusters are populated
        # keep = np.bincounts(delta) > 0 
        # reindex those clusters
        keep, delta[:] = np.unique(delta, return_inverse = True)
        # return new indices, cluster parameters associated with populated clusters
        return delta, zeta[keep]

    def sample_zeta_new(self, alpha, beta, m):
        return gamma(shape = alpha, scale = 1 / beta, size = (m, self.nCol + self.nCat))

    def sample_alpha(self, zeta, curr_alpha):
        n = zeta.shape[0]
        zs = zeta.sum(axis = 0)
        lzs = np.log(zeta).sum(axis = 0)
        args = zip(
            curr_alpha, repeat(n), zs, lzs,
            repeat(self.priors.alpha.a), repeat(self.priors.alpha.b),
            repeat(self.priors.beta.a), repeat(self.priors.beta.b),
            )
        res = map(sample_gamma_shape_wrapper, args)
        return np.array(list(res))

    def sample_beta(self, zeta, alpha):
        n  = zeta.shape[0]
        zs = zeta.sum(axis = 0)
        As = n * alpha + self.priors.beta.a
        Bs = zs + self.priors.beta.b
        beta = gamma(shape = As, scale = 1/Bs)
        beta[beta < 1e-9] = 1e-9
        return beta

    def sample_r(self, delta, zeta):
        # As = zeta[delta][:, :self.nCol].sum(axis = 1)
        # Bs = (self.data.Yp * sigma[delta][:, :self.nCol]).sum(axis = 1)
        As = np.einsum('il->i', zeta[:,:self.nCol][delta])
        Bs = np.einsum('il->i', self.data.Yp)
        return gamma(shape = As, scale = 1/Bs)

    def sample_rho(self, delta, zeta):
        """ Sampling the PG_1 gammas for categorical variables

        Args:
            delta ([type]): [description]
            zeta ([type]): [description]
        """
        As = zeta[:, self.nCol:][delta] + self.data.W
        Bs = 1.
        rho = gamma(shape = As, scale = 1 / Bs)
        rho[rho < 1e-9] = 1e-9
        return rho

    def sample_eta(self, curr_eta, delta):
        g = beta(curr_eta + 1, self.nDat)
        aa = self.priors.eta.a + delta.max() + 1
        bb = self.priors.beta.b - log(g)
        eps = (aa - 1) / (self.nDat  * bb + aa - 1)
        aaa = choice((aa, aa - 1), 1, p = (eps, 1 - eps))
        return gamma(shape = aaa, scale = 1 / bb)

    def sample_zeta(self, curr_zeta, r, rho, delta, alpha, beta):
        dmat = delta[:,None] == np.arange(delta.max() + 1) # n x J
        Y  = np.hstack((r[:, None] * self.data.Yp, rho)) # n x D
        lY = np.log(Y)
        W  = np.hstack((np.zeros(self.data.Yp.shape), self.data.W))

        lp_curr = np.zeros(curr_zeta.shape)
        lp_prop = np.zeros(curr_zeta.shape)

        curr_log_zeta = np.log(curr_zeta)
        prop_log_zeta = curr_log_zeta + normal(size = curr_log_zeta.shape, scale = 0.3)

        for i in range(curr_log_zeta.shape[1]):
            lp_curr.T[i] = log_post_log_zeta_1(curr_log_zeta.T[i], Y.T[i], lY.T[i], W.T[i], alpha[i], beta[i], dmat)
            lp_prop.T[i] = log_post_log_zeta_1(prop_log_zeta.T[i], Y.T[i], lY.T[i], W.T[i], alpha[i], beta[i], dmat)
        
        lp_diff = lp_prop - lp_curr
        keep = np.log(uniform(size = lp_curr.shape)) < lp_diff   # metropolis hastings step

        log_zeta = (prop_log_zeta * keep) + (curr_log_zeta * (~keep))
        return np.exp(log_zeta)

    def cluster_log_likelihood(self, zeta):
        out  = np.zeros((self.nDat, self.max_clust_count))
        out += dprojresgamma_log_my_mt(self.data.Yp, zeta.T[0:self.nCol].T)
        zstart = self.nCol
        wstart = 0
        for catlen in self.data.Cats:
            out += dprgmultinom_log_mw_mt(
                self.data.W.T[wstart:(wstart + catlen)].T.astype(int),
                zeta.T[zstart:(zstart + catlen)].T
                )
            zstart += catlen
            wstart += catlen
        return out

    def initialize_sampler(self, ns):
        self.samples = Samples(ns, self.nDat, self.nCol, self.nCat, self.nCats)
        self.samples.alpha[0] = 1.
        self.samples.beta[0] = 1.
        self.samples.zeta[0] = gamma(
                shape = 2., scale = 2., 
                size = (self.max_clust_count - 30, self.nCol + self.nCat),
                )
        self.samples.eta[0] = 40.
        self.samples.delta[0] = choice(self.max_clust_count - 30, size = self.nDat)
        self.samples.delta[0][-1] = np.arange(self.max_clust_count - 30)[-1]
        self.samples.r[0] = self.sample_r(
                self.samples.delta[0], self.samples.zeta[0],
                )
        self.samples.rho[0] = (self.CatMat / self.data.Cats).sum(axis = 1)
        self.curr_iter = 0
        self.sigma_placeholder = np.ones((self.max_clust_count, self.nCol + self.nCat))
        return

    def iter_sample(self):
        # current cluster assignments; number of new candidate clusters
        delta = self.curr_delta.copy();  m = self.max_clust_count - (delta.max() + 1)
        alpha = self.curr_alpha
        beta  = self.curr_beta
        zeta  = np.vstack((self.curr_zeta, self.sample_zeta_new(alpha, beta, m)))
        eta   = self.curr_eta
        r     = self.curr_r
        rho   = self.curr_rho

        self.curr_iter += 1
        # log_likelihood = dprodgamma_log_my_mt(
        #     np.hstack((r[:,None] * self.data.Yp, rho)), zeta, self.sigma_placeholder,
        #     )
        log_likelihood = self.cluster_log_likelihood(zeta)
        # pre-generate uniforms to inverse-cdf sample cluster indices
        unifs = uniform(size = self.nDat)
        # provide a cluster index probability placeholder, so it's not being re-allocated for every sample
        delta = diriproc_cluster_sampler(delta, log_likelihood, unifs, eta)
        # clean indices (clear out dropped clusters, unused candidate clusters, and re-index)
        delta, zeta = self.clean_delta_zeta(delta, zeta)
        self.samples.delta[self.curr_iter] = delta
        self.samples.r[self.curr_iter]     = self.sample_r(self.curr_delta, zeta)
        self.samples.rho[self.curr_iter]   = self.sample_rho(self.curr_delta, zeta)
        self.samples.zeta[self.curr_iter]  = self.sample_zeta(
                zeta, self.curr_r, self.curr_rho, self.curr_delta, alpha, beta,
                )
        self.samples.alpha[self.curr_iter] = self.sample_alpha(self.curr_zeta, alpha)
        self.samples.beta[self.curr_iter]  = self.sample_beta(self.curr_zeta, self.curr_alpha)
        self.samples.eta[self.curr_iter]   = self.sample_eta(eta, self.curr_delta)
        return

    def sample(self, ns):
        self.initialize_sampler(ns)
        print_string = '\rSampling {:.1%} Completed, {} Clusters     '
        print(print_string.format(self.curr_iter / ns, self.nDat), end = '')
        while (self.curr_iter < ns):
            if (self.curr_iter % 100) == 0:
                print(print_string.format(self.curr_iter / ns, self.curr_delta.max() + 1), end = '')
            self.iter_sample()
        print('\rSampling 100% Completed                    ')
        return

    def write_to_disk(self, path, nBurn, nThin = 1):
        folder = os.path.split(path)[0]
        if not os.path.exists(folder):
            os.mkdir(folder)
        if os.path.exists(path):
            os.remove(path)

        zetas  = np.vstack([
            np.hstack((np.ones((zeta.shape[0], 1)) * i, zeta))
            for i, zeta in enumerate(self.samples.zeta[nBurn :: nThin])
            ])
        rhos   = self.samples.rho[nBurn::nThin].reshape(-1, self.nCat)
        alphas = self.samples.alpha[nBurn :: nThin]
        betas  = self.samples.beta[nBurn :: nThin]
        deltas = self.samples.delta[nBurn :: nThin]
        rs     = self.samples.r[nBurn :: nThin]
        etas   = self.samples.eta[nBurn :: nThin]

        out = {
            'zetas'  : zetas,
            'alphas' : alphas,
            'betas'  : betas,
            'rhos'   : rhos,
            'rs'     : rs,
            'deltas' : deltas,
            'etas'   : etas,
            'nCol'   : self.nCol,
            'nDat'   : self.nDat,
            'nCat'   : self.nCat,
            'cats'   : self.data.Cats,
            'V'      : self.data.V,
            'W'      : self.data.W,
            }
        
        try:
            out['Y'] = self.data.Y
        except AttributeError:
            pass
        
        with open(path, 'wb') as file:
            pickle.dump(out, file)

        return

    def set_projection(self):
        self.data.Yp = (self.data.V.T / (self.data.V**self.p).sum(axis = 1)**(1/self.p)).T
        self.data.Yp[self.data.Yp <= 1e-6] = 1e-6
        return
    
    def categorical_considerations(self):
        """ Builds the CatMat """
        cats = np.hstack(list(np.ones(ncat) * i for i, ncat in enumerate(self.data.Cats)))
        self.CatMat = cats[:, None] == np.arange(len(self.data.Cats))
        return
    
    def __init__(
            self,
            data,
            prior_eta   = GammaPrior(2., 0.5),
            prior_alpha = GammaPrior(0.5, 0.5),
            prior_beta  = GammaPrior(0.5, 0.5),
            p           = 10,
            max_clust_count = 300,
            ):
        assert type(data) is MixedData
        self.data = data
        self.max_clust_count = max_clust_count
        self.p = p
        self.nCat = self.data.nCat
        self.nCol = self.data.nCol
        self.nDat = self.data.nDat
        self.nCats = self.data.Cats.shape[0]
        self.priors = Prior(prior_eta, prior_alpha, prior_beta)
        self.set_projection()
        self.categorical_considerations()
        # self.pool = Pool(processes = 8, initializer = limit_cpu())
        return

class Result(object):
    def generate_posterior_predictive_gammas(self, n_per_sample = 1, m = 10):
        new_gammas = []
        for s in range(self.nSamp):
            dmax = self.samples.delta[s].max()
            njs = np.bincount(self.samples.delta[s], minlength = int(dmax + 1 + m))
            ljs = njs + (njs == 0) * self.samples.eta[s] / m
            new_zetas = gamma(
                shape = self.samples.alpha[s],
                scale = 1. / self.samples.beta[s],
                size = (m, self.nCol + self.nCat),
                )
            prob = ljs / ljs.sum()
            deltas = cu.generate_indices(prob, n_per_sample)
            zeta = np.vstack((self.samples.zeta[s], new_zetas))[deltas]
            new_gammas.append(gamma(shape = zeta))
        return np.vstack(new_gammas)

    def generate_posterior_predictive_hypercube(self, n_per_sample = 1, m = 10):
        gammas = self.generate_posterior_predictive_gammas(n_per_sample, m)
        # hypercube transformation for real variates
        hypcube = euclidean_to_hypercube(gammas[:,:self.nCol])
        # simplex transformation for categ variates
        simplex_reverse = []
        indices = list(np.arange(self.nCol + self.nCat))
        # Foe each category, last first
        for i in list(range(self.cats.shape[0]))[::-1]:
            # identify the ending index (+1 to include boundary)
            cat_length = self.cats[i]
            cat_end = indices.pop() + 1
            # identify starting index
            for _ in range(cat_length - 1):
                cat_start = indices.pop()
            # transform gamma variates to simplex
            simplex_reverse.append(euclidean_to_simplex(gammas[:,cat_start:cat_end]))
        # stack hypercube and categorical variables side by side.
        return np.hstack([hypcube] + simplex_reverse[::-1])

    def generate_posterior_predictive_angular(self, n_per_sample = 1, m = 10):
        hyp = self.generate_posterior_predictive_hypercube(n_per_sample, m)
        return euclidean_to_angular(hyp)

    def write_posterior_predictive(self, path, n_per_sample = 1):
        colnames_y = ['Y_{}'.format(i) for i in range(self.nCol)]
        colnames_p = [
            ['p_{}_{}'.format(i,j) for j in range(catlength)]
            for i, catlength in enumerate(self.cats)
            ]
        colnames_p = list(chain(*colnames_p))

        thetas = pd.DataFrame(
                self.generate_posterior_predictive_hypercube(n_per_sample),
                # self.generate_posterior_predictive_angular(n_per_sample),
                #columns = ['theta_{}'.format(i) for i in range(1, self.nCol)],
                columns = colnames_y + colnames_p
                )
        thetas.to_csv(path, index = False)
        return

    def load_data(self, path):        
        with open(path, 'rb') as file:
            out = pickle.load(file)
        
        deltas = out['deltas']
        etas   = out['etas']
        zetas  = out['zetas']
        alphas = out['alphas']
        betas  = out['betas']
        rs     = out['rs']
        rhos   = out['rhos']
        cats   = out['cats']
        
        self.nSamp  = deltas.shape[0]
        self.nDat   = deltas.shape[1]
        self.nCat   = rhos.shape[1]
        self.nCol   = alphas.shape[1] - self.nCat
        self.nCats  = cats.shape[0]
        self.nSigma = self.nCol + self.nCat - self.nCats - 1
        self.cats   = cats
        self.data = BaseData(out['V'], out['W'])
        if 'Y' in out.keys():
            self.data.Y = out['Y']
        
        
        self.samples       = Samples(self.nSamp, self.nDat, self.nCol, self.nCat, self.nCats)
        self.samples.delta = deltas
        self.samples.eta   = etas
        self.samples.alpha = alphas
        self.samples.beta  = betas
        self.samples.zeta  = [zetas[np.where(zetas.T[0] == i)[0], 1:] for i in range(self.nSamp)]
        self.samples.r     = rs
        self.samples.rho   = rhos.reshape(self.nSamp, self.nDat, self.nCat)
        return

    def __init__(self, path):
        self.load_data(path)
        return

if __name__ == '__main__':
    from data import MixedData
    from projgamma import GammaPrior
    from pandas import read_csv
    import os

    raw = read_csv('./datasets/ad2_cover_x.csv')
    data = MixedData(raw, cat_vars = np.array([0,3], dtype = int), decluster = False, quantile = 0.999)
    model = Chain(data, prior_eta = GammaPrior(2, 1), p = 10)
    model.sample(4000)
    model.write_to_disk('./test/results.pickle', 2000, 2)
    res = Result('./test/results.pickle')
    res.write_posterior_predictive('./test/postpred.csv')

# EOF
