from numpy.random import choice, gamma, beta, normal, uniform
from collections import namedtuple
from itertools import repeat, chain
import numpy as np
import pandas as pd
import os
import sqlite3 as sql
from math import ceil, log
# from multiprocessing import Pool
from energy import limit_cpu

from scipy.special import gammaln


import cUtility as cu
from cProjgamma import sample_alpha_1_mh, sample_alpha_k_mh, sample_beta_fc, \
                        logddirichlet, logdgamma, logdgamma_restricted
from data import euclidean_to_angular, euclidean_to_simplex, euclidean_to_hypercube, Data
from projgamma import GammaPrior, DirichletPrior

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
    out -= np.einsum('jd,nd->nj', aBeta, aY)                          # e^(- beta y)
    return out

def dprodgamma_log_multi_y(aY, vAlpha, vBeta, logConstant):
    """ log density -- product of gammas -- multiple y's (array) against single theta (vector) """
    ld = (
        + logConstant
        # + (vAlpha * np.log(vBeta)).sum()
        # - gammaln(vAlpha).sum()
        + ((vAlpha - 1) * np.log(aY)).sum(axis = 1)
        - (vBeta * aY).sum(axis = 1)
        )
    return ld

def dprodgamma_log_multi_theta(vY, aAlpha, aBeta, vlogConstant):
    """ log density -- product of gammas -- single y (vector) against multiple thetas (matrix) """
    ld = (
        + vlogConstant 
        # + (aAlpha * np.log(aBeta)).sum(axis = 1)
        # - gammaln(aAlpha).sum(axis = 1)
        + np.dot(np.log(vY), (aAlpha - 1).T)
        - np.dot(vY, aBeta.T)
        # + (np.log(vY) * (aAlpha - 1)).sum(axis = 1)
        # - (vY * aBeta).sum(axis = 1)
        )
    return ld

def dgamma_log_multi_y(vY, alpha, beta, logConstant):
    """ log density -- gamma -- multiple y's (vector) against single theta (singlular) """
    ld = (
        + logConstant
        # + (alpha * log(beta))
        # - gammaln(alpha)
        + ((alpha - 1) * np.log(vY)).sum()
        - (beta * vY).sum()
        )
    return ld

def dgamma_log_multi_theta(y, vAlpha, vBeta, vlogConstant):
    """ log density -- gamma -- single y (singular) against multiple theta (vector) """
    ld = (
        + vlogConstant
        # + vAlpha * log(beta)
        # - gammaln(vAlpha)
        + (vAlpha - 1) * log(y)
        - (vBeta * y)
        )
    return ld

def update_zeta_j_wrapper(args):
    # parse arguments
    curr_zeta_j, Y_j, alpha, beta = args
    prop_zeta_j = np.empty(curr_zeta_j.shape)
    prop_zeta_j[0] = sample_alpha_1_mh(curr_zeta_j[0], Y_j.T[0], alpha[0], beta[0])
    for i in range(1, curr_zeta_j.shape[0]):
        prop_zeta_j[i] = sample_alpha_1_mh(
                curr_zeta_j[i], Y_j.T[i], alpha[i], beta[i],
                )
    return prop_zeta_j

def update_alpha_l_wrapper(args):
    return sample_alpha_k_mh(*args)

def update_beta_l_wrapper(args):
    return sample_beta_fc(*args)

Prior = namedtuple('Prior', 'eta alpha beta xi tau')

class Samples(object):
    zeta  = None
    alpha = None
    beta  = None
    delta = None
    r     = None
    eta   = None

    def __init__(self, nSamp, nDat, nCol):
        self.zeta  = [None] * (nSamp + 1)
        self.alpha = np.empty((nSamp + 1, nCol))
        self.beta  = np.empty((nSamp + 1, nCol))
        self.delta = np.empty((nSamp + 1, nDat), dtype = int)
        self.r     = np.empty((nSamp + 1, nDat))
        self.eta   = np.empty(nSamp + 1)
        return

class Chain(object):
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
        scratch *= 0        
        curr_cluster_state[delta_i] -= 1
        scratch += curr_cluster_state
        scratch += cand_cluster_state * (eta / (cand_cluster_state.sum() + 1e-9))
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            np.log(scratch, out = scratch)
        # scratch += np.log(curr_cluster_state + cand_cluster_state * eta / cand_cluster_state.sum())
        scratch += log_likelihood_i
        scratch -= scratch.max()
        np.exp(scratch, out = scratch)
        np.cumsum(scratch, out = scratch)
        delta_i = np.searchsorted(scratch, p * scratch[-1])
        curr_cluster_state[delta_i] += 1
        cand_cluster_state[delta_i] = False
        return delta_i

    # def sample_delta_i(self, curr_cluster_state, cand_cluster_state, 
    #                     eta, delta_i, p, y, zeta, logConstant, temp):
    #     """ 
    #     curr_cluster_state : vector of current cluster counts (np.bincount, integer)
    #     cand_cluster_state : vector of bools whether a given cluster is candidate
    #     eta   : candidate cluster weighting factor
    #     p     : random uniform (scalar between 0 and 1)
    #     y     : observation i (vector of length d)
    #     zeta  : shape matrix for gamma distribution (J x d)
    #     sigma : weight matrix for gamma distribution (J x d)
    #     """
    #     curr_cluster_state[delta_i] -= 1
    #     with np.errstate(divide = 'ignore', invalid = 'ignore'):
    #         temp[:] = (
    #             + np.log(curr_cluster_state + 
    #                 (cand_cluster_state * eta / cand_cluster_state.sum()))
    #             + dprodgamma_log_multi_theta(y, zeta, np.ones(self.nCol), logConstant)
    #             )
    #     temp -= temp.max()
    #     np.exp(temp, out = temp)
    #     np.cumsum(temp, out = temp)
    #     # temp[:] = np.exp(temp).cumsum()
    #     # temp /= temp[-1]
    #     delta_i = np.searchsorted(temp, p * temp[-1])
    #     curr_cluster_state[delta_i] += 1
    #     cand_cluster_state[delta_i] = False
    #     return delta_i
    
    def clean_delta_zeta(self, delta, zeta):
        """
        delta : cluster indicator vector (n)
        zeta  : cluster parameter matrix (J* x d)
        sigma : cluster parameter matrix (J* x d)
        """
        # which clusters are populated
        # keep = np.bincounts(delta) > 0 
        # reindex those clusters
        keep, delta[:] = np.unique(delta, return_inverse = True)
        # return new indices, cluster parameters associated with populated clusters
        return delta, zeta[keep]

    def sample_zeta_new(self, alpha, beta, m):
        return gamma(shape = alpha, scale = 1/beta, size = (m, self.nCol))

    def sample_alpha(self, zeta, curr_alpha):
        args = zip(
            curr_alpha,
            zeta.T,
            repeat(self.priors.alpha.a),
            repeat(self.priors.alpha.b),
            repeat(self.priors.beta.a),
            repeat(self.priors.beta.b),
            )
        res = map(update_alpha_l_wrapper, args)
        # res = self.pool.map(update_alpha_l_wrapper, args)
        return np.array(list(res))

    def sample_beta(self, zeta, alpha):
        args = zip(
            alpha,
            zeta.T,
            repeat(self.priors.beta.a),
            repeat(self.priors.beta.b),
            )
        res = map(update_beta_l_wrapper, args)
        # res = self.pool.map(update_beta_l_wrapper, args)
        return np.array(list(res))

    def sample_r(self, delta, zeta):
        As = zeta[delta].sum(axis = 1)
        Bs = np.ones(self.nCol) @ self.data.Yp.T
        # Bs = (self.data.Yp * sigma[delta]).sum(axis = 1)
        return gamma(shape = As, scale = 1/Bs)

    def sample_eta(self, curr_eta, delta):
        g = beta(curr_eta + 1, self.nDat)
        aa = self.priors.eta.a + delta.max() + 1
        bb = self.priors.beta.b - log(g)
        eps = (aa - 1) / (self.nDat  * bb + aa - 1)
        aaa = choice((aa, aa - 1), 1, p = (eps, 1 - eps))
        return gamma(shape = aaa, scale = 1 / bb)

    def sample_zeta(self, curr_zeta, r, delta, alpha, beta):
        Y = r.reshape(-1, 1) * self.data.Yp
        args = zip(
            curr_zeta,
            [Y[np.where(delta == j)[0]] for j in range(curr_zeta.shape[0])],
            repeat(alpha),
            repeat(beta)
            )
        res = map(update_zeta_j_wrapper, args)
        # res = self.pool.map(update_zeta_j_wrapper, args)
        return np.array(list(res))

    def initialize_sampler(self, ns):
        self.samples = Samples(ns, self.nDat, self.nCol)
        self.samples.alpha[0] = 1.
        self.samples.beta[0] = 1.
        self.samples.zeta[0] = gamma(shape = 2., scale = 2., size = (self.max_clust_count - 5, self.nCol))
        self.samples.eta[0] = 40.
        self.samples.delta[0] = choice(self.max_clust_count - 5, size = self.nDat)
        self.samples.r[0] = self.sample_r(self.samples.delta[0], self.samples.zeta[0])
        self.curr_iter = 0
        return

    def iter_sample(self):
        # current cluster assignments; number of new candidate clusters
        delta = self.curr_delta.copy();  m = self.max_clust_count - (delta.max() + 1)
        curr_cluster_state = np.bincount(delta, minlength = self.max_clust_count)
        cand_cluster_state = np.hstack((np.zeros(delta.max() + 1, dtype = bool), np.ones(m, dtype = bool)))
        alpha = self.curr_alpha
        beta  = self.curr_beta
        zeta  = np.vstack((self.curr_zeta, self.sample_zeta_new(alpha, beta, m)))
        eta   = self.curr_eta
        r     = self.curr_r

        self.curr_iter += 1
        # normalizing constant for product of Gammas
        # logConstant = - gammaln(zeta).sum(axis = 1)
        log_likelihood = dprodgamma_log_my_mt(r.reshape(-1,1) * self.data.Yp, zeta, np.ones(zeta.shape))
        # pre-generate uniforms to inverse-cdf sample cluster indices
        unifs   = uniform(size = self.nDat)
        # provide a cluster index probability placeholder, so it's not being re-allocated for every sample
        scratch = np.empty(self.max_clust_count)
        for i in range(self.nDat):
            delta[i] = self.sample_delta_i(
                    curr_cluster_state, cand_cluster_state, eta,
                    log_likelihood[i], delta[i], unifs[i], scratch,
                    )
            # Sample new cluster indices
            # delta[i] = self.sample_delta_i(curr_cluster_state, cand_cluster_state, eta, 
            #                 delta[i], unifs[i], r[i] * self.data.Yp[i], zeta, logConstant, scratch)
        # clean indices (clear out dropped clusters, unused candidate clusters, and re-index)
        delta, zeta = self.clean_delta_zeta(delta, zeta)
        self.samples.delta[self.curr_iter] = delta
        self.samples.r[self.curr_iter]     = self.sample_r(self.curr_delta, zeta)
        self.samples.zeta[self.curr_iter]  = self.sample_zeta(
                zeta, self.curr_r, self.curr_delta, alpha, beta,
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
            if (self.curr_iter % 10) == 0:
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
        conn = sql.connect(path)

        zetas  = np.vstack([
            np.hstack((np.ones((zeta.shape[0], 1)) * i, zeta))
            for i, zeta in enumerate(self.samples.zeta[nBurn :: nThin])
            ])
        alphas = self.samples.alpha[nBurn :: nThin]
        betas  = self.samples.beta[nBurn :: nThin]
        deltas = self.samples.delta[nBurn :: nThin]
        rs     = self.samples.r[nBurn :: nThin]
        etas   = self.samples.eta[nBurn :: nThin]

        zetas_df = pd.DataFrame(
                zetas, columns = ['iter'] + ['zeta_{}'.format(i) for i in range(self.nCol)],
                )
        alphas_df = pd.DataFrame(alphas, columns = ['alpha_{}'.format(i) for i in range(self.nCol)])
        betas_df  = pd.DataFrame(betas,  columns = ['beta_{}'.format(i)  for i in range(self.nCol)])
        deltas_df = pd.DataFrame(deltas, columns = ['delta_{}'.format(i) for i in range(self.nDat)])
        rs_df     = pd.DataFrame(rs,     columns = ['r_{}'.format(i)     for i in range(self.nDat)])
        etas_df   = pd.DataFrame({'eta' : etas})

        zetas_df.to_sql('zetas',   conn, index = False)
        alphas_df.to_sql('alphas', conn, index = False)
        betas_df.to_sql('betas',   conn, index = False)
        deltas_df.to_sql('deltas', conn, index = False)
        rs_df.to_sql('rs',         conn, index = False)
        etas_df.to_sql('etas',     conn, index = False)
        conn.commit()
        conn.close()
        return

    def set_projection(self):
        self.data.Yp = (self.data.V.T / (self.data.V**self.p).sum(axis = 1)**(1/self.p)).T
        return

    def __init__(
            self,
            data,
            prior_eta   = GammaPrior(2., 0.5),
            prior_alpha = GammaPrior(0.5, 0.5),
            prior_beta  = GammaPrior(2., 2.),
            prior_xi    = GammaPrior(0.5, 0.5),
            prior_tau   = GammaPrior(2., 2.),
            p           = 10,
            max_clust_count = 300,
            ):
        self.data = data
        self.max_clust_count = max_clust_count
        self.p = p
        self.nCol = self.data.nCol
        self.nDat = self.data.nDat
        self.priors = Prior(prior_eta, prior_alpha, prior_beta, prior_xi, prior_tau)
        self.set_projection()
        # self.pool = Pool(processes = 8, initializer = limit_cpu())
        return

class Result(object):
    def generate_posterior_predictive_gammas(self, n_per_sample = 1, m = 10):
        new_gammas = []
        for s in range(self.nSamp):
            dmax = self.samples.delta[s].max()
            njs = np.bincount(self.samples.delta[s], int(dmax + 1 + m))
            ljs = njs + (njs == 0) * self.samples.eta[s] / m
            new_zetas = gamma(
                shape = self.samples.alpha[s],
                scale = 1. / self.samples.beta[s],
                size = (m, self.nCol),
                )
            prob = ljs / ljs.sum()
            deltas = cu.generate_indices(prob, n_per_sample)
            zeta = np.vstack((self.samples.zeta[s], new_zetas))[deltas]
            new_gammas.append(gamma(shape = zeta))
        return np.vstack(new_gammas)

    def generate_posterior_predictive_hypercube(self, n_per_sample = 1, m = 10):
        gammas = self.generate_posterior_predictive_gammas(n_per_sample, m)
        return euclidean_to_hypercube(gammas)

    def generate_posterior_predictive_angular(self, n_per_sample = 1, m = 10):
        hyp = self.generate_posterior_predictive_hypercube(n_per_sample, m)
        return euclidean_to_angular(hyp)

    def write_posterior_predictive(self, path, n_per_sample = 1):
        thetas = pd.DataFrame(
                self.generate_posterior_predictive_angular(n_per_sample),
                columns = ['theta_{}'.format(i) for i in range(1, self.nCol)],
                )
        thetas.to_csv(path, index = False)
        return

    def load_data(self, path):
        conn = sql.connect(path)

        deltas = pd.read_sql('select * from deltas;', conn).values.astype(int)
        etas   = pd.read_sql('select * from etas;', conn).values
        zetas  = pd.read_sql('select * from zetas;', conn).values
        alphas = pd.read_sql('select * from alphas;', conn).values
        betas  = pd.read_sql('select * from betas;', conn).values
        rs     = pd.read_sql('select * from rs;', conn).values

        self.nSamp = deltas.shape[0]
        self.nDat  = deltas.shape[1]
        self.nCol  = alphas.shape[1]

        self.samples       = Samples(self.nSamp, self.nDat, self.nCol)
        self.samples.delta = deltas
        self.samples.eta   = etas
        self.samples.alpha = alphas
        self.samples.beta  = betas
        self.samples.zeta  = [zetas[np.where(zetas.T[0] == i)[0], 1:] for i in range(self.nSamp)]
        self.samples.r     = rs
        return

    def __init__(self, path):
        self.load_data(path)
        self.data = Data(os.path.join(os.path.split(path)[0], 'empirical.csv'))
        return

# EOF


if __name__ == '__main__':
    from data import Data_From_Raw
    from projgamma import GammaPrior
    from pandas import read_csv
    import os

    raw = read_csv('./datasets/ivt_nov_mar.csv')
    data = Data_From_Raw(raw, decluster = True, quantile = 0.95)
    data.write_empirical('./test/empirical.csv')
    model = Chain(data, prior_eta = GammaPrior(2, 1), p = 10)
    model.sample(10000)
    model.write_to_disk('./test/results.db', 5000, 5)
    res = Result('./test/results.db')
    res.write_posterior_predictive('./test/postpred.csv')
    # EOL

# EOF 2