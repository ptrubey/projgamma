import numpy as np
import numpy.typing as npt
from numpy.random import choice, gamma, uniform, normal
from scipy.stats import invwishart
from numpy.linalg import cholesky, inv
from collections import namedtuple

# np.seterr(invalid='raise')
EPS = np.finfo(float).eps

from .samplers import ParallelTemperingStickBreakingSampler,                     \
    bincount2D_vectorized, pt_py_sample_chi_bgsb, pt_py_sample_cluster_bgsb,    \
    NormalPrior, InvWishartPrior, GEMPrior
from .data import Projection, Data, category_matrix, euclidean_to_catprob,       \
    euclidean_to_hypercube, euclidean_to_psphere, euclidean_to_simplex
from .densities import pt_logd_cumdircategorical_mx_ma_inplace_unstable,         \
    pt_logd_mvnormal_mx_st, logd_mvnormal_mx_st, logd_invwishart_ms,            \
    pt_logd_cumdirmultinom_paired_yt, pt_logd_projgamma_my_mt_inplace_unstable, \
    pt_logd_projgamma_paired_yt, pt_logd_pareto_mx_ma_inplace_unstable,         \
    pt_logd_pareto_paired_yt
from .cov import PerObsTemperedOnlineCovariance

Prior = namedtuple('Prior', 'mu Sigma chi')

class Samples(object):
    zeta  = None
    mu    = None
    Sigma = None
    delta = None
    chi   = None

    def to_dict(self, nBurn : int = 0, nThin : int = 1) -> dict:
        out = {
            'zeta'  : self.zeta[(nBurn+1)  :: nThin, 0],
            'mu'    : self.mu[(nBurn+1)    :: nThin, 0],
            'Sigma' : self.Sigma[(nBurn+1) :: nThin, 0],
            'delta' : self.delta[(nBurn+1) :: nThin, 0], 
            'chi'   : self.chi[(nBurn+1)   :: nThin, 0],
            }
        return out

    @classmethod
    def from_dict(cls, out):
        return cls(**out)

    @classmethod
    def from_meta(
            cls, 
            nSamp : int, 
            nDat : int, 
            tCol : int, 
            nTemp : int, 
            nTrunc : int
            ):
        zeta  = np.empty((nSamp + 1, nTemp, nTrunc, tCol))
        mu    = np.empty((nSamp + 1, nTemp, tCol))
        Sigma = np.empty((nSamp + 1, nTemp, tCol, tCol))
        delta = np.empty((nSamp + 1, nTemp, nDat), dtype = int)
        chi   = np.empty((nSamp + 1, nTemp, nTrunc))
        return cls(zeta, mu, Sigma, delta, chi)

    def __init__(
            self, 
            zeta  : npt.NDArray[np.float64], 
            mu    : npt.NDArray[np.float64], 
            Sigma : npt.NDArray[np.float64], 
            delta : npt.NDArray[np.int32], 
            chi   : npt.NDArray[np.float64],
            ):
        self.zeta  = zeta
        self.mu    = mu
        self.Sigma = Sigma
        self.delta = delta
        self.chi   = chi
        return

class Samples_(Samples):
    def __init__(self, nSamp, nDat, tCol, nTrunc):
        self.zeta  = np.empty((nSamp, nTrunc, tCol))
        self.mu    = np.empty((nSamp, tCol))
        self.Sigma = np.empty((nSamp, tCol, tCol))
        self.delta = np.empty((nSamp, nDat))
        self.chi   = np.empty((nSamp, nTrunc))
        return

class Chain(ParallelTemperingStickBreakingSampler, Projection):
    @property
    def curr_zeta(self) -> npt.NDArray[np.float64]:
        return self.samples.zeta[self.curr_iter]
    @property
    def curr_mu(self) -> npt.NDArray[np.float64]:
        return self.samples.mu[self.curr_iter]
    @property
    def curr_Sigma(self) -> npt.NDArray[np.float64]:
        return self.samples.Sigma[self.curr_iter]
    @property
    def curr_delta(self) -> npt.NDArray[np.int32]:
        return self.samples.delta[self.curr_iter]
    @property
    def curr_chi(self) -> npt.NDArray[np.float64]:
        return self.samples.chi[self.curr_iter]
    
    # Adaptive Metropolis Placeholders
    am_Sigma : PerObsTemperedOnlineCovariance = None
    am_scale : float = None
    max_clust_count : int = None
    swap_attempts : npt.NDArray[np.int32] = None
    swap_succeeds : npt.NDArray[np.int32] = None

    def sample_delta(
            self, 
            chi : npt.NDArray[np.float64], 
            zeta : npt.NDArray[np.float64],
            ) -> npt.NDArray[np.float64]:
        log_likelihood = self.log_delta_likelihood(zeta)
        delta = pt_py_sample_cluster_bgsb(chi, log_likelihood)
        return delta

    def sample_zeta_new(
            self, 
            mu : npt.NDArray[np.float64], 
            Sigma_chol : npt.NDArray[np.float64],
            ) -> npt.NDArray[np.float64]:
        """ Sample new zetas as log-normal (sample normal, then exponentiate) """
        sizes = (self.nTemp, self.max_clust_count, self.tCol)
        out = np.empty(sizes)
        np.einsum('tzy,tjy->tjz', np.triu(Sigma_chol), normal(size = sizes), out = out)
        out += mu[:,None,:]
        np.exp(out, out=out)
        return out

    def am_covariance_matrices(
            self, 
            delta : npt.NDArray[np.int32], 
            index : npt.NDArray[np.int32],
            ) -> npt.NDArray[np.float64]:
        return self.am_Sigma.cluster_covariance(delta)[index]

    def sample_zeta(
            self, 
            zeta : npt.NDArray[np.float64], 
            delta : npt.NDArray[np.int32], 
            mu : npt.NDArray[np.float64], 
            Sigma_chol : npt.NDArray[np.float64], 
            Sigma_inv : npt.NDArray[np.float64],
            ) -> npt.NDArray[np.float64]:
        """
        zeta      : (t x J x D)
        delta     : (t x n)
        r         : (t x n)
        mu        : (t x D)
        Sigma_cho : (t x D x D)
        Sigma_inv : (t x D x D)
        """
        curr_cluster_state = bincount2D_vectorized(delta, self.max_clust_count)
        cand_cluster_state = (curr_cluster_state == 0)
        delta_ind_mat = delta[:,:,None] == range(self.max_clust_count)
        idx = np.where(~cand_cluster_state)
        covs = self.am_covariance_matrices(delta, idx)
        
        am_alpha = np.zeros((self.nTemp, self.max_clust_count))
        am_alpha[:] = -np.inf
        am_alpha[idx] = 0.

        zcurr = zeta.copy()
        with np.errstate(divide = 'ignore'):
            lzcurr = np.log(zeta)
        lzcand = lzcurr.copy()
        lzcand[idx] += np.einsum(
            'mpq,mq->mp', 
            cholesky(self.am_scale * covs), 
            normal(size = (idx[0].shape[0], self.tCol)),
            )
        zcand = np.exp(lzcand)
        
        am_alpha += self.log_zeta_likelihood(zcand, delta, delta_ind_mat)
        am_alpha -= self.log_zeta_likelihood(zcurr, delta, delta_ind_mat)
        with np.errstate(invalid = 'ignore'):
            am_alpha *= self.itl[:,None]
        am_alpha += self.log_logzeta_prior(lzcand, mu, Sigma_chol, Sigma_inv)
        am_alpha -= self.log_logzeta_prior(lzcurr, mu, Sigma_chol, Sigma_inv)
        
        keep = np.where(np.log(uniform(size = am_alpha.shape)) < am_alpha)
        zcurr[keep] = zcand[keep]
        return zcurr

    def sample_Sigma(
            self, 
            zeta : npt.NDArray[np.float64], 
            mu : npt.NDArray[np.float64], 
            extant_clusters : npt.NDArray[np.bool_],
            ) -> npt.NDArray[np.float64]:
        n = extant_clusters.sum(axis = 1)
        diff = (np.log(zeta) - mu[:,None,:]) * extant_clusters[:,:,None]
        C = np.einsum('tjd,tje->tde', diff, diff)
        _psi = self.priors.Sigma.psi + C * self.itl[:,None,None]
        _nu  = self.priors.Sigma.nu + n * self.itl
        out = np.empty((self.nTemp, self.tCol, self.tCol))
        for i in range(self.nTemp):
            out[i] = invwishart.rvs(df = _nu[i], scale = _psi[i])
        return out

    def sample_mu(
            self, 
            zeta : npt.NDArray[np.float64], 
            Sigma_inv : npt.NDArray[np.float64], 
            extant_clusters : npt.NDArray[np.bool_],
            ) -> npt.NDArray[np.float64]:
        n = extant_clusters.sum(axis = 1)
        assert np.all(zeta[extant_clusters] > 0)
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            lzbar = np.nansum(np.log(zeta) * extant_clusters[:,:,None], axis = 1) / n[:,None]
        _Sigma = inv(n[:,None,None] * Sigma_inv + self.priors.mu.SInv)
        _mu = np.einsum(
            'tjl,tl->tj', 
            _Sigma, 
            self.priors.mu.SInv @ self.priors.mu.mu + 
                np.einsum('tjl,tl->tj', Sigma_inv, n[:,None] * lzbar),
            )
        out = np.zeros((self.nTemp, self.tCol))
        np.einsum(
            'tkl,tl->tk', cholesky(_Sigma), 
            normal(size = (self.nTemp, self.tCol)),
            out = out,
            )
        out += _mu
        return out

    def sample_chi(
            self, 
            delta : npt.NDArray[np.int32],
            ) -> npt.NDArray[np.float64]:
        chi = pt_py_sample_chi_bgsb(
            delta,
            *self.priors.chi,
            trunc = self.max_clust_count,
            )
        return chi

    def log_delta_likelihood(
            self, 
            zeta : npt.NDArray[np.float64],
            ) -> npt.NDArray[np.float64]:
        out = np.zeros((self.nDat, self.nTemp, self.max_clust_count))
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            pt_logd_projgamma_my_mt_inplace_unstable(
                out, self.data.Yp , zeta[:,:,:self.nCol], self.sigma_ph1,
                )
            pt_logd_cumdircategorical_mx_ma_inplace_unstable(
                out, self.data.W, zeta[:,:,self.nCol:(self.nCol + self.nCat)], self.CatMat,
                )
            if self.model_radius:
                pt_logd_pareto_mx_ma_inplace_unstable(
                    out, self.data.R, zeta[:,:,-1],
                    )
        np.nan_to_num(out, False, -np.inf)
        return out
    
    def log_zeta_likelihood(
            self, 
            zeta : npt.NDArray[np.float64], 
            delta : npt.NDArray[np.int32], 
            delta_ind_mat : npt.NDArray[np.bool_],
            ) -> npt.NDArray[np.float64]:
        out = np.zeros((self.nTemp, self.max_clust_count))
        zetas = zeta[
            self.temp_unravel, delta.ravel(),
            ].reshape(self.nTemp, self.nDat, self.tCol)
        out += np.einsum(
            'tn,tnj->tj',
            pt_logd_projgamma_paired_yt(
                self.data.Yp, zetas[:,:,:self.nCol], self.sigma_ph2,
                ),
            delta_ind_mat,
            )
        out += np.einsum(
            'tn,tnj->tj',
            pt_logd_cumdirmultinom_paired_yt(
                self.data.W, zetas[:,:,self.nCol:(self.nCol + self.nCat)], self.CatMat,
                ),
            delta_ind_mat,
            )
        if self.model_radius:
            out += np.einsum(
                'tn,tnj->tj',
                pt_logd_pareto_paired_yt(self.data.R, zetas[:,:,-1]),
                delta_ind_mat,
                )
        return out
    
    def log_logzeta_prior(
            self, 
            logzeta : npt.NDArray[np.float64], 
            mu : npt.NDArray[np.float64], 
            Sigma_chol : npt.NDArray[np.float64], 
            Sigma_inv : npt.NDArray[np.float64],
            ) -> npt.NDArray[np.float64]:
        """
        logzeta   :  (t, j, d)
        mu        :  (t, d)
        Sigma_inv :  (t,d,d)
        """
        return pt_logd_mvnormal_mx_st(logzeta, mu, Sigma_chol, Sigma_inv)

    def log_tempering_likelihood(self) -> npt.NDArray[np.float64]:
        curr_zeta = self.curr_zeta[
            self.temp_unravel, self.curr_delta.ravel()
            ].reshape(self.nTemp, self.nDat, self.tCol)
        out = np.zeros(self.nTemp)
        out += pt_logd_projgamma_paired_yt(
            self.data.Yp, 
            curr_zeta[:,:,:self.nCol],
            self.sigma_ph2,
            ).sum(axis = 1)
        out += pt_logd_cumdirmultinom_paired_yt(
            self.data.W, 
            curr_zeta[:,:,self.nCol:(self.nCol + self.nCat)],
            self.CatMat,
            ).sum(axis = 1)
        if self.model_radius:
            out += pt_logd_pareto_paired_yt(self.data.R, curr_zeta[:,:,-1]).sum(axis = 1)
        return out

    def log_tempering_prior(self) -> npt.NDArray[np.float64]:
        out = np.zeros(self.nTemp)
        Sigma_cho = cholesky(self.curr_Sigma)
        Sigma_inv = inv(self.curr_Sigma)
        extant_clusters = (bincount2D_vectorized(self.curr_delta, self.max_clust_count) > 0)
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            out += np.nansum(
                extant_clusters * pt_logd_mvnormal_mx_st(
                    np.log(self.curr_zeta), self.curr_mu, Sigma_cho, Sigma_inv,
                    ),
                axis = 1,
                )
        out += logd_mvnormal_mx_st(self.curr_mu, *self.priors.mu)
        out += logd_invwishart_ms(self.curr_Sigma, *self.priors.Sigma)
        return out

    def initialize_sampler(self, ns : int) -> None:
        # Samples
        self.samples = Samples(ns, self.nDat, self.tCol, self.nTemp, self.max_clust_count)
        self.samples.mu[0] = 0.
        self.samples.Sigma[0] = np.eye(self.tCol) * 2.
        self.samples.zeta[0] = gamma(
                shape = 2., scale = 2., 
                size = (self.nTemp, self.max_clust_count, self.tCol),
                )
        self.samples.chi[0] = 0.1
        self.samples.delta[0] = choice(
                self.max_clust_count - 50, 
                size = (self.nTemp, self.nDat),
                )
        # Iterator
        self.curr_iter = 0
        # Adaptive Metropolis related
        self.swap_attempts = np.zeros((self.nTemp, self.nTemp))
        self.swap_succeeds = np.zeros((self.nTemp, self.nTemp))
        # Placeholders
        self.sigma_ph1 = np.ones((self.nTemp, self.max_clust_count, self.nCol))
        self.sigma_ph2 = np.ones((self.nTemp, self.nDat, self.nCol))
        self.zeta_shape = (self.nTemp, self.nDat, self.tCol)
        return

    def update_am_cov(self) -> None:
        """ Online updating for Adaptive Metropolis Covariance per obsv. """
        lzeta = np.swapaxes(
            np.log(
                self.curr_zeta[
                    self.temp_unravel, self.curr_delta.ravel()
                    ].reshape(
                        self.nTemp, self.nDat, self.tCol
                        )
                ),
            0, 1,
            )
        self.am_Sigma.update(lzeta)
        return

    def try_tempering_swap(self) -> None:
        ci = self.curr_iter
        # declare log-likelihood, log-prior
        lpl = self.log_tempering_likelihood()
        lpp = self.log_tempering_prior()
        # declare swap choices
        sw  = choice(self.nTemp, 2 * self.nSwap_per, replace = False).reshape(-1, 2)
        for s in sw:
            # record attempted swap
            self.swap_attempts[s[0],s[1]] += 1
            self.swap_attempts[s[1],s[0]] += 1
        # compute swap log-probability
        sw_alpha = np.zeros(sw.shape[0])
        sw_alpha += lpl[sw.T[1]] - lpl[sw.T[0]]
        sw_alpha *= self.itl[sw.T[1]] - self.itl[sw.T[0]]
        sw_alpha += lpp[sw.T[1]] - lpp[sw.T[0]]
        logp = np.log(uniform(size = sw_alpha.shape))
        for tt in sw[np.where(logp < sw_alpha)[0]]:
            # report successful swap
            self.swap_succeeds[tt[0],tt[1]] += 1
            self.swap_succeeds[tt[1],tt[0]] += 1
            # do the swap
            self.samples.zeta[ci][tt[0]], self.samples.zeta[ci][tt[1]] =   \
                self.samples.zeta[ci][tt[1]].copy(), self.samples.zeta[ci][tt[0]].copy()
            self.samples.mu[ci][tt[0]], self.samples.mu[ci][tt[1]] = \
                self.samples.mu[ci][tt[1]].copy(), self.samples.mu[ci][tt[0]].copy()
            self.samples.Sigma[ci][tt[0]], self.samples.Sigma[ci][tt[1]] = \
                self.samples.Sigma[ci][tt[1]].copy(), self.samples.Sigma[ci][tt[0]].copy()
            self.samples.delta[ci][tt[0]], self.samples.delta[ci][tt[1]] = \
                self.samples.delta[ci][tt[1]].copy(), self.samples.delta[ci][tt[0]].copy()
        return

    def iter_sample(self) -> None:
        # current cluster assignments; number of new candidate clusters
        delta = self.curr_delta.copy()
        zeta  = self.curr_zeta.copy()
        mu    = self.curr_mu
        Sigma = self.curr_Sigma
        Sigma_cho = cholesky(self.curr_Sigma)
        Sigma_inv = inv(Sigma)
        chi   = self.curr_chi

        # Adaptive Metropolis Update
        self.update_am_cov()
        
        # Advance the iterator
        self.curr_iter += 1
        ci = self.curr_iter

        # Sample new candidate clusters
        cluster_state = bincount2D_vectorized(delta, self.max_clust_count)
        cand_clusters = np.where(cluster_state == 0)
        zeta[cand_clusters] = self.sample_zeta_new(mu, Sigma_cho)[cand_clusters]
        
        # Update cluster assignments and re-index
        self.samples.delta[ci] = self.sample_delta(chi, zeta)
        self.samples.chi[ci] = self.sample_chi(self.curr_delta)
        
        # do rest of sampling
        extant_clusters = (cluster_state > 0)
        self.samples.zeta[ci] = self.sample_zeta(
            zeta, self.curr_delta, mu, Sigma_cho, Sigma_inv,
            )
        self.samples.mu[ci] = self.sample_mu(zeta, Sigma_inv, extant_clusters)
        self.samples.Sigma[ci] = self.sample_Sigma(zeta, mu, extant_clusters)

        # Attempt Swap:
        if self.curr_iter >= self.swap_start:
           self.try_tempering_swap()
        return

    def to_dict(self, nBurn : int = 0, nThin : int = 1) -> dict:
        out = {
            'data'    : self.data.to_dict(),
            'samples' : self.samples.to_dict(),
            'priors'  : self.priors,
            'time'    : self.time_elapsed_numeric,
            'swap_y'  : self.swap_succeeds,
            'swap_p'  : self.swap_succeeds / (self.swap_attempts + 1e-9),
            'model_radius' : self.model_radius,
            }
        return out
    
    def categorical_considerations(self) -> None:
        """ Builds the CatMat """
        self.CatMat = category_matrix(self.data.cats)
        return
    
    def __init__(
            self,
            data,
            prior_mu     = NormalPrior(0, 1.),
            prior_Sigma  = InvWishartPrior(100, 1.),
            prior_chi    = (0.1, 1.),
            p            = 10,
            max_clust_count = 200,
            ntemps       = 3,
            stepping     = 1.1,
            model_radius = False,
            **kwargs
            ):
        assert type(data) is Data
        if model_radius:
            assert type(data.R) is np.ndarray
        self.model_radius = model_radius
        self.data = data
        self.max_clust_count = max_clust_count
        self.p = p
        self.nCat = self.data.nCat
        self.nCol = self.data.nCol
        self.tCol = self.nCol + self.nCat
        if self.model_radius:
            self.tCol += 1
        self.nDat = self.data.nDat
        self.nCats = self.data.cats.shape[0]

        # Setting Priors
        _prior_mu = NormalPrior(
            np.ones(self.tCol) * prior_mu[0], 
            np.eye(self.tCol) * np.sqrt(prior_mu[1]),
            np.eye(self.tCol) / np.sqrt(prior_mu[1]),            
            )
        nu = self.tCol + prior_Sigma[0]
        psi = np.eye(self.tCol) * nu * prior_Sigma[1]
        psi = np.zeros((self.tCol, self.tCol))
        psi[:self.nCol, :self.nCol] = np.eye(self.nCol)
        start_idx = self.nCol
        for catlength in self.data.cats:
            end_idx = start_idx + catlength
            cat_cov = np.eye(end_idx - start_idx)
            for i in range(catlength):
                for j in range(catlength):
                    if i != j:
                        cat_cov[i,j] = - catlength**(-2.)
            psi[start_idx:end_idx, start_idx:end_idx] = cat_cov
            start_idx = end_idx
        if self.model_radius:
            psi[-1,-1] = 1.
        _prior_Sigma = InvWishartPrior(nu, psi)
        psi *= nu * prior_Sigma[1]
        _prior_chi = GEMPrior(*prior_chi)
        self.priors = Prior(_prior_mu, _prior_Sigma, _prior_chi)
        self.set_projection()
        self.categorical_considerations()

        # Parallel Tempering
        self.nTemp = ntemps
        self.itl = 1 / stepping**np.arange(ntemps)
        self.temp_unravel = np.repeat(np.arange(self.nTemp), self.nDat)
        self.nSwap_per = self.nTemp // 2
        self.swap_start = 100

        # Adaptive Metropolis
        self.am_Sigma = PerObsTemperedOnlineCovariance(
            self.nTemp, self.nDat, self.tCol, self.max_clust_count
            )
        self.am_scale = 2.38**2 / self.tCol
        return

class Result(object):
    priors : Prior[NormalPrior, InvWishartPrior, GEMPrior]

    def generate_posterior_predictive_gammas(
            self, 
            n_per_sample : int = 1, 
            m : int = 10,
            ) -> npt.NDArray[np.float64]:
        zetas = [] 
        njs = np.zeros(self.max_clust_count, dtype = int)
        ljs = np.zeros(self.max_clust_count)
        prob = np.zeros(self.max_clust_count)
        for s in range(self.nSamp):
            njs[:] = np.bincount(self.samples.delta[s], minlength = self.max_clust_count)
            ljs[:] = 0.
            ljs += njs
            ljs -= (njs > 0) * self.priors.chi.discount
            ljs += (njs == 0) * self.priors.chi.concentration / m
            ljs += (njs == 0) * self.priors.chi.discount * (njs > 0).sum() / m
            prob[:] = ljs / ljs.sum()
            Sprob = np.cumsum(prob, axis = -1)
            unis  = uniform(size = n_per_sample)
            delta = np.searchsorted(Sprob, unis)
            zetas.append(self.samples.zeta[s][delta])
        zetas = np.vstack(zetas)
        return gamma(shape = zetas)

    def generate_posterior_predictive_hypercube(
            self, 
            n_per_sample : int = 1, 
            m : int = 10,
            ) -> npt.NDArray[np.float64]:
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

    def generate_posterior_predictive_spheres(self, n_per_sample : int) -> npt.NDArray[np.float64]:
        rhos = self.generate_posterior_predictive_gammas(n_per_sample)[:,self.nCol:(self.nCol + self.nCat)] # (s,D)
        CatMat = category_matrix(self.data.Cats) # (C,d)
        return euclidean_to_catprob(rhos, CatMat)
        
    def generate_conditional_posterior_predictive_radii(self) -> npt.NDArray[np.float64]:
        """ r | zeta, V ~ Gamma(r | sum(zeta), sum(V)) """
        shapes = np.array([
            zeta[delta][:,:self.nCol]
            for delta, zeta
            in zip(self.samples.delta, self.samples.zeta)
            ]).sum(axis = 2)
        rates = self.data.V.sum(axis = 1)[None,:]
        rs = gamma(shape = shapes, scale = 1 / rates)
        return rs

    def generate_conditional_posterior_predictive_gammas(self) -> npt.NDArray[np.float64]:
        """ rho | zeta, delta + W ~ Gamma(rho | zeta[delta] + W) """
        zetas = np.swapaxes(np.array([
            zeta[delta]
            for delta, zeta 
            in zip(self.samples.delta, self.samples.zeta)
            ]),0,1) # (n,s,d)
        W = np.hstack((np.zeros((self.nDat, self.nCol)), self.data.W)) # (n,d)
        return gamma(shape = zetas + W[:,None,:])

    def generate_conditional_posterior_predictive_spheres(self) -> npt.NDArray[np.float64]:
        """ pi | zeta, delta = normalized rho
        currently discarding generated Y's, keeping latent pis
        """
        rhos = self.generate_conditional_posterior_predictive_gammas()[:,:,self.nCol:(self.nCol + self.nCat)]
        CatMat = category_matrix(self.data.Cats) # (C,d)
        shro = rhos @ CatMat.T # (s,n,C)
        nrho = np.einsum('snc,cd->snd', shro, CatMat) # (s,n,d)
        pis = rhos / nrho
        return pis

    def generate_new_conditional_posterior_predictive_spheres(
            self, 
            Vnew : npt.NDArray[np.float64], 
            Wnew : npt.NDArray[np.bool_], 
            Rnew : npt.NDArray[np.float64],
            ) -> npt.NDArray[np.float64]:
        rhos   = self.generate_new_conditional_posterior_predictive_gammas(
            Vnew, Wnew, Rnew,
            )[:,:,self.nCol:]
        CatMat = category_matrix(self.data.Cats)
        shro   = rhos @ CatMat.T
        nrho   = np.einsum('snc,cd->snd', shro, CatMat) # (s,n,d)
        pis    = rhos / nrho
        return pis
    
    def generate_new_conditional_posterior_predictive_radii(
            self, 
            Vnew : npt.NDArray[np.float64], 
            Wnew : npt.NDArray[np.bool_], 
            Rnew : npt.NDArray[np.float64],
            ) -> npt.NDArray[np.float64]:
        znew = self.generate_new_conditional_posterior_predictive_zetas(Vnew, Wnew, Rnew)
        shapes = znew[:,:,:self.nCol].sum(axis = 2)
        return gamma(shapes)
    
    def generate_new_conditional_posterior_predictive_gammas(
            self, 
            Vnew : npt.NDArray[np.float64], 
            Wnew : npt.NDArray[np.int32], 
            Rnew : npt.NDArray[np.float64],
            ) -> npt.NDArray[np.float64]:
        znew = self.generate_new_conditional_posterior_predictive_zetas(Vnew, Wnew, Rnew)
        return gamma(znew)

    def generate_new_conditional_posterior_predictive_hypercube(
            self, 
            Vnew : npt.NDArray[np.float64], 
            Wnew : npt.NDArray[np.bool_], 
            Rnew : npt.NDArray[np.float64],
            ) -> npt.NDArray[np.float64]:
        znew = self.generate_new_conditional_posterior_predictive_zetas(Vnew, Wnew, Rnew)
        Ypnew = euclidean_to_psphere(Vnew, 10)
        R = gamma(znew[:,:,:self.nCol].sum(axis = 2))
        G = gamma(znew[:,:,self.nCol:(self.nCol + self.nCat)])
        return euclidean_to_hypercube(np.hstack((R[:,:,None] * Ypnew, G)))
    
    def generate_new_conditional_posterior_predictive_euclidean(
            self, 
            Vnew : npt.NDArray[np.float64], 
            Wnew : npt.NDArray[np.bool_], 
            Rnew : npt.NDArray[np.float64],
            ) -> npt.NDArray[np.float64]:
        znew = self.generate_new_conditional_posterior_predictive_zetas(Vnew, Wnew, Rnew)
        Ypnew = euclidean_to_psphere(Vnew, 10)
        R = gamma(znew[:,:,:self.nCol].sum(axis = 2))
        G = gamma(znew[:,:,self.nCol:(self.nCol + self.nCat)])
        return np.hstack((R[:,:,None] * Ypnew, G))

    def generate_new_conditional_posterior_predictive_zetas(
            self, 
            Vnew : npt.NDArray[np.float64], 
            Wnew : npt.NDArray[np.bool_], 
            Rnew : npt.NDArray[np.float64],
            ) -> npt.NDArray[np.float64]:
        n = Vnew.shape[0]
        Ypnew = euclidean_to_psphere(Vnew, 10)
        weights = np.zeros((self.nSamp, self.max_clust_count))
        for s in range(self.nSamp):
            weights[s] = np.bincount(self.samples.delta[s], minlength = self.max_clust_count)

        weights -= (weights > 0) * self.GEMPrior.discount
        weights += (weights == 0) * (
            + self.GEMPrior.concentration
            + self.GEMPrior.discount * (weights > 0).sum(axis = 1)[:,None]
            ) / ((weights == 0).sum(axis = 1) + EPS)[:,None]
        np.log(weights, out = weights)
        loglik = np.zeros((n, self.nSamp, self.max_clust_count))
        sigma_ph = np.ones((1, self.max_clust_count, self.nCol))
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            pt_logd_projgamma_my_mt_inplace_unstable(
                loglik, Ypnew , self.samples.zeta[:,:,:self.nCol], sigma_ph,
                )
            pt_logd_cumdircategorical_mx_ma_inplace_unstable(
                loglik, Wnew, self.samples.zeta[:,:,self.nCol:(self.nCol + self.nCat)], self.CatMat,
                )
            if self.model_radius:
                pt_logd_pareto_mx_ma_inplace_unstable(loglik, Rnew, self.samples.zeta[:,:,-1])
        np.nan_to_num(loglik, False, -np.inf)
        # combine logprior weights and likelihood under cluster
        weights = weights[None] + loglik # (n, nSamp, maxclustcount)
        weights -= weights.max(axis = 2)[:,:,None]
        np.exp(weights, out = weights) # unnormalized cluster probability
        np.cumsum(weights, axis = 2, out = weights)
        weights /= weights[:,:,-1][:,:,None]  # normalized cluster cumulative probability
        p = uniform(size = (n, self.nSamp, 1))
        dnew = (p > weights).sum(axis = 2)  # new deltas
        znew = np.empty((n, self.nSamp, self.tCol))
        for i in range(n):
            for s in range(self.nSamp):
                znew[i,s] = self.samples.zeta[s,dnew[i,s]]
        return znew
    
    def load_data(self, out : dict) -> None:        
        self.samples = Samples.from_dict(out['samples'])
        self.data = Data.from_dict(out['data'])
        self.priors = out['priors']
        self.model_radius = out['model_radius']
        self.time = out['time']
        self.swap_y = out['swap_y']
        self.swap_p = out['swap_p']
        
        self.nSamp, self.nDat = self.samples.delta.shape
        self.nCat = self.data.nCat
        self.nCol = self.data.nCol
        self.tCol = self.nCol + self.nCat
        if self.model_radius: 
            self.tCol += 1
        self.nCats  = self.data.cats.shape[0]
        self.cats   = self.data.cats
        self.CatMat = category_matrix(self.data.cats)
        self.max_clust_count = self.samples.chi.shape[-1] + 1
        return

    def __init__(self, out : dict):
        self.load_data(out)
        return

if __name__ == '__main__':
    from data import Data
    from densities import GammaPrior
    from pandas import read_csv
    import os

    # p = argparser()
    # d = {
    #     'in_data_path'      : './ad/solarflare/cat_data.csv',
    #     'in_outcome_path'   : './ad/solarflare/cat_outcome.csv',
    #     'out_path'          : './ad/solarflare/temp_cat_results.pkl',
    #     # 'in_data_path'    : './ad/cardio/data_new.csv',
    #     # 'in_outcome_path' : './ad/cardio/outcome_new.csv',
    #     # 'out_path' : './ad/cardio/results_test.pkl',
    #     # 'cat_vars' : '[5,6,7,8,9]',
    #     'realtype' : 'rank',
    #     'cat_vars' : '[0,1,2,3,4,5,6,7,8,9]',
    #     'decluster' : 'False',
    #     'quantile' : 0.95,
    #     'nSamp' : 2000,
    #     'nKeep' : 1000,
    #     'nThin' : 2,
    #     'eta_alpha' : 2.,
    #     'eta_beta' : 1.,
    #     }
    # p = Heap(**d)
    class Heap(object):
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            return
    d = {
        'in_data_path' : './datasets/ivt_nov_mar.csv',
        'out_path'     : './test/results.pkl',
        'cat_vars'     :  '[]',
        'realtype'     : 'threshold',
        'decluster'    : 'True',
        'quantile'     : 0.95,
        'nSamp'        : 10000,
        'nThin'        : 2,
        'nKeep'        : 8000,
    }
    p = Heap(**d)

    raw = read_csv(p.in_data_path).values
    raw = raw[~np.isnan(raw).any(axis = 1)]
    # out = read_csv(p.in_outcome_path).values
    # out = out[~np.isnan(out).any(axis = 1)].ravel()
    # assert raw.shape[0] == out.shape[0]
    # data = MixedData(
    #     raw, 
    #     cat_vars = np.array(eval(p.cat_vars), dtype = int), 
    #     realtype = p.realtype,
    #     decluster = eval(p.decluster), 
    #     quantile = float(p.quantile),
    #     # outcome = out,
    #     )
    # data.fill_outcome(out)
    from data import Data
    data = Data.from_raw(
        raw,
        xh1t_cols = np.arange(raw.shape[1]),
        dcls = True,
        xhquant = 0.95,
        )
    model = Chain(
        data, prior_chi = GEMPrior(0.1, 0.1), p = 10, ntemps = 6,
        )
    model.sample(p.nSamp, verbose = True)
    model.write_to_disk(p.out_path, p.nKeep, p.nThin)
    res = Result(p.out_path)
    print(res.samples.zeta.max())
    print(res.samples.zeta.min())
    print(res.samples.zeta.mean())

    # Y,V,W,R = res.data.to_mixed_new(raw, out)
    # from anomaly import ResultFactory
    # res = ResultFactory('pypprgln', p.out_path)
    # Y,V,W,R = res.data.to_mixed_new(raw, out)
    # res.pools_open()
    # scores = res.get_scoring_metrics(Y,V,W,R)
    # res.pools_closed()

    # ppg = res.generate_posterior_predictive_gammas()
    # cppg = res.generate_new_conditional_posterior_predictive_gammas(V, W, R)
    # res.write_posterior_predictive('./test/postpred.csv')

# EOF
