""" 
Functions relating to density of Projected Gamma.  All functions are
parameterized such that E(x) = alpha / beta (treat beta as rate parameter). 
"""
import numpy as np

np.seterr(under = 'ignore', over = 'raise')
from collections import namedtuple
from contextlib import nullcontext
from functools import lru_cache
from math import acos, cos, exp, log, sin

from numpy.linalg import inv, norm, slogdet
from scipy.integrate import quad
from scipy.special import gammainc, gammaln, multigammaln, loggamma, softmax,   \
    log_softmax
from scipy.stats import gamma
from scipy.stats import norm as normal
from scipy.stats import uniform

from genpareto import gpd_fit

# Tuples for storing priors

GammaPrior     = namedtuple('GammaPrior', 'a b')
DirichletPrior = namedtuple('DirichletPrior', 'a')
BetaPrior      = namedtuple('BetaPrior', 'a b')
UniNormalPrior = namedtuple('UniNormalPrior','mu sigma')
InvGammaPrior  = namedtuple('InvGammaPrior', 'a b')
# LogNormal Models
NormalPrior     = namedtuple('NormalPrior', 'mu SCho SInv')
InvWishartPrior = namedtuple('InvWishartPrior', 'nu psi')

## Functions related to projected gamma density

def logd_gamma(vY, alpha, beta):
    out = np.errstate(vY.shape[0])
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        out += alpha * np.log(beta)
        out -= loggamma(alpha)
        out += (alpha - 1) * np.log(vY)
        out -= beta * vY
    np.nan_to_num(out, False, -np.inf)
    return out

def logd_prodgamma_my_mt(aY, aAlpha, aBeta):
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
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        out += np.einsum('jd,jd->j', aAlpha, np.log(aBeta)).reshape(1,-1) # beta^alpha
        out -= np.einsum('jd->j', loggamma(aAlpha)).reshape(1,-1)          # gamma(alpha)
        out += np.einsum('jd,nd->nj', aAlpha - 1, np.log(aY))             # y^(alpha - 1)
        out -= np.einsum('jd,nd->nj', aBeta, aY)                          # e^(- beta y)
    np.nan_to_num(out, False, -np.inf)
    return out

def pt_logd_prodgamma_my_mt(aY, aAlpha, aBeta):
    """
    product of gammas log-density for multiple y, multiple theta 
    ----
    aY     : array of Y     (t x n x d) [Y in R^d]
    aAlpha : array of alpha (t x J x d)
    aBeta  : array of beta  (t x J x d)
    ----
    returns: (n x t x J)
    """
    t, n, d = aY.shape; j = aAlpha.shape[1] # set dimensions
    ld = np.zeros((n, t, j))
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        ld += np.einsum('tjd,tjd->tj', aAlpha, np.log(aBeta)).reshape(1, t, j)
        ld -= np.einsum('tjd->tj', loggamma(aAlpha)).reshape(1, t, j)
        ld += np.einsum('tnd,tjd->ntj', np.log(aY), aAlpha - 1)
        ld -= np.einsum('tnd,tjd->ntj', aY, aBeta)
    np.nan_to_num(ld, False, -np.inf)
    return ld

def logd_prodgamma_paired(aY, aAlpha, aBeta):
    """
    product of gammas log-density for paired y, theta
    ----
    aY     : array of Y     (n x d) [Y in R^d]
    aAlpha : array of alpha (n x d)
    aBeta  : array of beta  (n x d)
    ----
    returns: (n)
    """
    out = np.zeros(aY.shape[0])
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        out += np.einsum('nd,nd->n', aAlpha, np.log(aBeta))    # beta^alpha
        out -= np.einsum('nd->n', loggamma(aAlpha))            # gamma(alpha)
        out += np.einsum('nd,nd->n', np.log(aY), (aAlpha - 1)) # y^(alpha - 1)
        out -= np.einsum('nd,nd->n', aY, aBeta)                # e^(-y beta)
    np.nan_to_num(out, False, -np.inf)
    return out

def pt_logd_prodgamma_paired(aY, aAlpha, aBeta):
    """
    product of gammas log-density for paired y, theta
    ----
    aY     : array of Y     (t x n x d) [Y in R^d]
    aAlpha : array of alpha (t x n x d)
    aBeta  : array of beta  (t x n x d)
    ----
    returns: (t x n)
    """
    out = np.zeros(aY.shape[:-1])                             # n temps x n Y
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        out += np.einsum('tnd,tnd->tn', aAlpha, np.log(aBeta))    # beta^alpha
        out -= np.einsum('tnd->tn', loggamma(aAlpha))             # gamma(alpha)
        out += np.einsum('tnd,tnd->tn', np.log(aY), (aAlpha - 1)) # y^(alpha - 1)
        out -= np.einsum('tnd,tnd->tn', aY, aBeta)                # e^(-y beta)
    np.nan_to_num(out, False, -np.inf)
    return out                                     # per-temp,Y log-density

def logd_prodgamma_my_st(aY, vAlpha, vBeta):
    out = np.zeros(aY.shape[0])
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        out += (vAlpha * np.log(vBeta)).sum()  # scalar
        out -= sum(loggamma(vAlpha))           # scalar
        out += np.einsum('nd,d->n', np.log(aY), vAlpha - 1)
        out -= np.einsum('nd,d->n', aY, vBeta)
    np.nan_to_num(out, False, -np.inf)
    return out

def pt_logd_prodgamma_my_st(aY, aAlpha, aBeta):
    """
    Log-density for product of Gammas 
        for Multiple Y, single theta (per temperature)
    ----
    Inputs:
        aY      (t, n, d)
        aAlpha  (t, d)
        aBeta   (t, d)
    Returns:
        log-density (t, n)
    """
    ld = np.zeros(aY.shape[:-1])
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        ld += np.einsum('td,td->t', aAlpha, np.log(aBeta)).reshape(-1,1)
        ld -= np.einsum('td->t', loggamma(aAlpha)).reshape(-1,1)
        ld += np.einsum('tnd,td->tn', np.log(aY), aAlpha - 1)
        ld -= np.einsum('tnd,td->tn', aY, aBeta)
    np.nan_to_num(ld, False, -np.inf)
    return ld

def pt_logd_projgamma_my_mt(aY, aAlpha, aBeta):
    # def dprojgamma_log_my_mt(aY, aAlpha, aBeta):
    """
    projected gamma log-density (proportional) 
        for multiple Y, multiple theta (per temperature)
    ----
    Inputs:
        aY     : array of Y     (n, d)    [Y in S_p^{d-1}]
        aAlpha : array of alpha (t, j, d)
        aBeta  : array of beta  (t, j, d)
    Returns:
        log-density (n, t, j)
    """
    n = aY.shape[0]; t,j,d = aAlpha.shape # set dimensions
    ld = np.zeros((n,t,j))
    aSum = aAlpha.sum(axis = -1)
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        ld += np.einsum('tjd,tjd->tj', aAlpha, np.log(aBeta))[None]
        ld -= np.einsum('tjd->tj', loggamma(aAlpha))[None]
        ld += np.einsum('nd,tjd->ntj', np.log(aY), aAlpha - 1)
        ld += loggamma(aSum)[None]
        ld -= aSum[None] * np.log(np.einsum('nd,tjd->ntj', aY, aBeta))
    np.nan_to_num(ld, False, -np.inf)
    return ld

def pt_logd_projgamma_my_mt_inplace_unstable(out, aY, aAlpha, aBeta):
    """
    projected gamma log-density (proportional) 
        for multiple Y, multiple theta (per temperature)
    ----
    Inputs:
        out    : log-density    (n, t, j)
        aY     : array of Y     (n, d)    [Y in S_p^{d-1}]
        aAlpha : array of alpha (t, j, d)
        aBeta  : array of beta  (t, j, d)
    """
    if aY.shape[1] == 0:
        return
    Asum = aAlpha.sum(axis = -1)
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        out += np.einsum('tjd,tjd->tj', aAlpha, np.log(aBeta))[None]
        out -= np.einsum('tjd->tj', loggamma(aAlpha))[None]
        out += np.einsum('nd,tjd->ntj', np.log(aY), aAlpha - 1)
        out += loggamma(Asum)[None]
        out -= Asum[None] * np.log(np.einsum('nd,tjd->ntj', aY, aBeta))
    np.nan_to_num(out, False, -np.inf)
    return

def logd_projgamma_my_mt_inplace_unstable(out, aY, aAlpha, aBeta):
    if aY.shape[0] == 0:
        return
    Asum = aAlpha.sum(axis = -1)
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        out += np.einsum('jd,jd->j', aAlpha, np.log(aBeta))[None]
        out -= np.einsum('jd->j', loggamma(aAlpha))[None]
        out += np.einsum('nd,jd->nj', np.log(aY), aAlpha - 1)
        out += loggamma(Asum)[None]
        out -= Asum[None] * np.log(np.einsum('nd,jd->nj', aY, aBeta))
    np.nan_to_num(out, False, -np.inf)
    return

def pt_logd_projgamma_paired_yt(aY, aAlpha, aBeta):
    # def dprojgamma_log_paired_yt(aY, aAlpha, aBeta):
    """
    projected gamma log-density (proportional) for paired y, theta
    ----
    aY     : array of Y     (n, d) [Y in S_p^{d-1}]
    aAlpha : array of alpha (t, n, d)
    aBeta  : array of beta  (t, n, d)
    ----
    returns: (t, n)
    """
    if aAlpha.shape[-1] == 0:
        return np.zeros(aAlpha.shape[:-1])
    ld = np.zeros(aAlpha.shape[:-1])
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        ld += np.einsum('tnd,tnd->tn', aAlpha, np.log(aBeta))
        ld -= np.einsum('tnd->tn', loggamma(aAlpha))
        ld += np.einsum('nd,tnd->tn', np.log(aY), (aAlpha - 1))
        ld += loggamma(np.einsum('tnd->tn', aAlpha))
        ld -= np.einsum('tnd->tn',aAlpha) * np.log(np.einsum('nd,tnd->tn', aY, aBeta))
    np.nan_to_num(ld, False, -np.inf)
    return ld

def pt_logd_dirichlet_mx_ma(aY, aAlpha):
    """
    Log-density for Dirichlet Distribution (assuming parallel tempering)
    ---
    Args:
        aY     : (n, d)
        aAlpha : (t,j,d)
    ---
    Returns:
        out    : (n,t,j)
    """
    n = aY.shape[0]; t, j, d = aAlpha.shape
    out = np.zeros((n,t,j))
    out += loggamma(aAlpha.sum(axis = 2))
    out -= loggamma(aAlpha).sum(axis = 2)
    out += np.einsum('nd,tjd->ntj', np.log(aY), aAlpha - 1)
    return out

def pt_stickbreak(nu):
    """
    parallel tempered stickbreaking function.
    nu : (T x (J-1))
    out : (T x J)
    """
    out = np.zeros((nu.shape[0], nu.shape[1] + 1))
    out[..., :-1] += np.log(nu)
    out[..., 1: ] += np.cumsum(np.log(1 - nu), axis = -1)
    # np.exp(out, out = out)
    return softmax(out, axis = -1)

def pt_log_stickbreak(nu):
    """ Parallel Tempered stick-breaking function (log) """
    shape = list(nu.shape)
    shape[-1] += 1
    out = np.zeros(shape = shape)
    out[..., :-1] += np.log(nu)
    out[..., 1: ] += np.cumsum(np.log(1 - nu), axis = -1)
    return log_softmax(out, axis = -1)

def pt_logd_mixprojgamma(aY, aAlpha, aBeta, nu):
    """ 
    Parallel tempered mixture of projected gammas (omitting normalizing constant) 
    aY: (N x D)
    aAlpha : (T x J x D)
    nu : (T x J-1)    
    """
    N = aY.shape[0]
    T, J, D = aAlpha.shape
    
    logpi = pt_log_stickbreak(nu)
    scratch = np.zeros((aY.shape[0], *aAlpha.shape[:-1])) # (N, ..., J)
    pt_logd_projgamma_my_mt_inplace_unstable(scratch, aY, aAlpha, aBeta)
    scratch += logpi[None]
    np.exp(scratch, out = scratch)
    return np.log(scratch.sum(axis = -1))

## functions relating to gamma density

def logd_gamma_my(aY, alpha, beta):
    """
    log-density of Gamma distribution

    Args:
        aY    : (n x d)
        alpha : float
        beta  : float
    Returns: 
        ld    : (n x d)
    """
    lp = np.zeros(aY.shape)
    with np.errstate(divide = 'ignore',invalid = 'ignore'):
        lp += alpha * np.log(beta)
        lp -= loggamma(alpha)
        lp += (alpha - 1) * np.log(aY)
        lp -= beta * aY
    np.nan_to_num(lp, False, -np.inf)
    return lp

def pt_logd_gamma_my(aY, aAlpha, aBeta):
    """
    log-density of Gamma distribution
    
    Args:
        aY      : (t, j, d)
        aAlpha  : (t, d)
        aBeta   : (t, d)
    Returns:
        ld      : (t, j)
    """
    ld = np.zeros(aY.shape)
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        ld += (aAlpha * np.log(aBeta))[:, None]
        ld -= loggamma(aAlpha)[:, None]
        ld += (aAlpha - 1)[:, None] * np.log(aY)
        ld -= aBeta[:, None] * aY
    np.nan_to_num(ld, False, -np.inf)
    return ld.sum(axis = 2)

## functions relating to log-normal density
def pt_logd_lognormal_my(aY, aMu, aSigma):
    """
    log-density of Log-normal distribution
    ---
    Args:
        aY     : (t, j, d)
        aMu    : (t, d)
        aSigma : (t, d)
    Returns:
        ld     : (t, j)
    """
    logY = np.log(aY)
    diff = (np.log(aY) - aMu[:,None]) / aSigma[:,None]

    out = np.zeros(aY.shape)
    out -= logY
    out -= np.sqrt(2 * np.pi)
    out -= np.log(aSigma)[:,None]
    out -= 0.5 * diff * diff
    return out

def logd_lognormal_my(aY, mu, sigma):
    """
    log-density of log-normal distribution
    ---
    Args:
        aY     : (t, d)
        aMu    : scalar
        sigma  : scalar
    Returns: 
        ld     : (t, d)    
    """
    logY = np.log(aY)
    diff = (logY - mu) / sigma

    out = np.zeros(aY.shape)
    out -= logY
    out -= np.sqrt(2 * np.pi)
    out -= np.log(sigma)
    out -= 0.5 * diff * diff
    return out

def logd_normal_my(aY, mu, sigma):
    """
    log-density of log-normal distribution
    ---
    Args:
        aY     : (t, d)
        aMu    : scalar
        sigma  : scalar
    Returns: 
        ld     : (t, d)    
    """
    diff = (aY - mu) / sigma

    out = np.zeros(aY.shape)
    out -= np.sqrt(2 * np.pi)
    out -= np.log(sigma)
    out -= 0.5 * diff * diff
    return out

## Functions relating to MVnormal density

def pt_logd_mvnormal_mx_st(x, mu, cov_chol, cov_inv):
    # def dmvnormal_log_mx(x, mu, cov_chol, cov_inv):
    """ 
    multivariate normal log-density for multiple x, single theta per temp
    ------
    x        : array of alphas    (t x j x d)
    mu       : array of mus       (t x d)
    cov_chol : array of cov chols (t x d x d)
    cov_inv  : array of cov mats  (t x d x d)    
    """
    ld = np.zeros(x.shape[:-1])
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        ld -= np.einsum('tdd->t', np.log(cov_chol)).reshape(-1,1)
    ld -= 0.5 * np.einsum(
        'tjd,tdl,tjl->tj', 
        x - mu.reshape(-1,1,cov_chol.shape[1]), 
        cov_inv, 
        x - mu.reshape(-1,1,cov_chol.shape[1]),
        )
    np.nan_to_num(ld, False, -np.inf)
    return ld

def logd_mvnormal_mx_st(x, mu, cov_chol, cov_inv):
    #dmvnormal_log_mx_st(x, mu, cov_chol, cov_inv):
    ld = np.zeros(x.shape[:-1])
    ld -= np.log(np.diag(cov_chol)).sum()
    ld -= 0.5 * np.einsum('td,dl,tl->t',x - mu, cov_inv, x - mu)
    return ld

def logd_invwishart_ms(Sigma, nu, psi):
    # def dinvwishart_log_ms(Sigma, nu, psi):
    ld = np.zeros(Sigma.shape[0])
    ld += 0.5 * nu * slogdet(psi)[1]
    ld -= multigammaln(nu / 2, psi.shape[-1])
    ld -= 0.5 * nu * psi.shape[-1] * log(2.)
    ld -= 0.5 * (nu + Sigma.shape[-1] + 1) * slogdet(Sigma)[1]
    ld -= 0.5 * np.einsum(
            '...ii->...', np.einsum('ji,...ij->...ij', psi, inv(Sigma)),
            )
    return ld

## Functions relating to dirichlet-multinomial density

def logd_dirmultinom_mx_sa(aW, vAlpha):
    # def logd_dirichlet_multinomial_mx_sa(aW, vAlpha):
    """
    log-density for Dirichlet-Multinomial;
    calculates log-density for each observation (aW, axis 0)
    for one set of shape parameters
    (use for updating shape parameters per cluster)
    ----
    inputs:
        aW     : (n x d)
        aAlpha : (d)
    outputs:
        logd   : (n)
    """
    sa = vAlpha.sum()
    sw = aW.sum(axis = 1)
    logd = np.zeros(aW.shape[0])
    logd += loggamma(sa)
    logd += loggamma(sw + 1)
    logd -= loggamma(sw + sa)
    logd += loggamma(aW + vAlpha[None,:]).sum(axis = 1)
    logd -= loggamma(vAlpha).sum()
    logd -= loggamma(aW + 1).sum(axis = 1)
    return logd

def logd_dirmultinom_paired(aW, aAlpha):
    """
    log-density for Dirichlet-Multinomial
    calculates log-density for each observation/shape parameter pair
    ---
    inputs:
        aW     : (n x d)
        aAlpha : (n x d)
    """
    sa = aAlpha.sum(axis = 1)
    sw = aW.sum(axis = 1)
    logd = np.zeros(aW.shape[0])
    logd += loggamma(sa)
    logd += loggamma(sw + 1)
    logd -= loggamma(sw + sa)
    logd += loggamma(aW + aAlpha).sum(axis = 1)
    logd -= loggamma(aAlpha).sum(axis = 1)
    logd -= loggamma(aW + 1).sum(axis = 1)
    return logd

def logd_cumdirmultinom_mx_sa(aW, vAlpha, sphere_mat):
    sa = np.einsum('d,cd->c', vAlpha, sphere_mat)
    sw = np.einsum('nd,cd->nc', aW, sphere_mat)
    logd = np.zeros(aW.shape[0])
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        logd += loggamma(sa).sum()
        logd += np.einsum('nc->n', loggamma(sw + 1))
        logd -= np.einsum('nc->n', loggamma(sw + sa[None,:]))
        logd += np.einsum('nd->n', loggamma(aW + vAlpha[None,:]))
        logd -= loggamma(vAlpha).sum()
        logd -= np.einsum('nd->n', loggamma(aW + 1))
    np.nan_to_num(logd, False, -np.inf)
    return logd

def logd_cumdircateg_mx_sa(aW, vAlpha, sphere_mat):
    sa = np.einsum('d,cd->c', vAlpha, sphere_mat)
    sw = np.einsum('nd,cd->nc', aW, sphere_mat) # matrix of ones (n x c)
    logd = np.zeros(aW.shape[0])
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        logd += loggamma(sa).sum()
        # logd += np.einsum('nc->n', loggamma(sw + 1)) # zeros, all the way down
        # logd -= np.einsum('tnc->tn', loggamma(sw[None,:,:] + sa[:,None,:])) # loggamma(sa + 1)
        logd -= loggamma(sa + 1.).sum()
        logd += np.einsum('nd->n', loggamma(aW + vAlpha[None,:]))
        logd -= loggamma(vAlpha).sum()
        # logd -= np.einsum('nd->n', loggamma(aW + 1)) # zeros again
    np.nan_to_num(logd, False, -np.inf)
    return logd

def pt_logd_cumdirmultinom_mx_sa(aW, aAlpha, sphere_mat):
    sa = np.einsum('td,cd->tc', aAlpha, sphere_mat) # (t,c)
    sw = np.einsum('nd,cd->nc', aW, sphere_mat)     # (n,c)
    logd = np.zeros((aAlpha.shape[0], aW.shape[0])) # (t,n)
    with np.errstate(divide = 'ignore',  invalid = 'ignore'):
        logd += np.einsum('', loggamma(sa))[:,None] # (t,)
        logd += np.einsum('nc->n', loggamma(sw + 1))[None,:] # (,n) 
        logd -= np.einsum('tnc->tn', loggamma(sw[None,:,:] + sa[:,None,:])) # (t,n)
        logd += np.einsum('tnd->tn', loggamma(aW[None,:,:] + aAlpha[:,None,:])) # (t,n)
        logd -= np.einsum('td,t', loggamma(aAlpha))[:,None] # (t,)
        logd -= np.einsum('nd->n', loggamma(aW + 1))[None,:]
    np.nan_to_num(logd, False, -np.inf)
    return logd

def logd_cumdirmultinom_mx_ma(aW, aAlpha, sphere_mat):
    """
    Log-density of concatenated Dirichlet-Multinomial distribution
    ---
    Inputs: 
        aW         (n x d)
        aAlpha     (j x d)
        sphere_mat (c x d)
    Output:
        logd (n x j)
    """
    sa = np.einsum('jd,cd->jc', aAlpha, sphere_mat)
    sw = np.einsum('nd,cd->nc', aW, sphere_mat)
    logd = np.zeros((aW.shape[0], aAlpha.shape[0]))
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        logd += np.einsum('jc->j', loggamma(sa))[None,:]
        logd += np.einsum('nc->n', loggamma(sw + 1))[:,None]
        logd -= np.einsum('njc->nj', loggamma(sw[:,None,:] + sa[None,:,:]))
        logd += np.einsum('njd->nj', loggamma(aW[:,None,:] + aAlpha[None,:,:]))
        logd -= np.einsum('jd->j', loggamma(aAlpha))[None,:]
        logd -= np.einsum('nd->n', loggamma(aW + 1))[:,None]
    np.nan_to_num(logd, False, -np.inf)
    return logd

def pt_logd_cumdirmultinom_mx_ma(aW, aAlpha, sphere_mat):
    """
    inputs:
        aW:         (n,d)
        aAlpha:     (t,j,d)
        sphere_mat: (c,d)
    output:
        logd:       (n,t,j)
    """
    sa = np.einsum('tjd,cd->tjc', aAlpha, sphere_mat)
    sw = np.einsum('nd,cd->nc', aW, sphere_mat)
    logd = np.zeros((aW.shape[0], aAlpha.shape[0], aAlpha.shape[1]))
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        logd += np.einsum('tjc->tj', loggamma(sa))[None,:,:]
        logd += np.einsum('nc->n', loggamma(sw + 1))[:,None,None]
        logd -= np.einsum('tnjc->ntj', loggamma(sw[None,:,None,:] + sa[:,None,:,:])) # (n,c)+(tjc)->(tnjc)
        logd += np.einsum('tnjd->ntj',loggamma(aW[None,:,None,:] + aAlpha[:,None,:,:])) # (nd)+(tjd)->(tnjd)
        logd -= np.einsum('tjd->tj', loggamma(aAlpha))[None,:,:]
        logd -= np.einsum('nd->n', loggamma(aW + 1))[:,None,None]
    np.nan_to_num(logd, False, -np.inf)
    return logd

def pt_logd_cumdircategorical_mx_ma_inplace_unstable_older(out, aW, aAlpha, sphere_mat):
    sa = np.einsum('tjd,cd->tjc', aAlpha, sphere_mat)
    sw = np.einsum('nd,cd->nc', aW, sphere_mat)
    out += np.einsum('tjc->tj', loggamma(sa))[None,:,:]
    out -= np.einsum('tnjc->ntj', loggamma(sw[None,:,None,:] + sa[:,None,:,:])) # (n,c)+(tjc)->(tnjc)
    out += np.einsum('tnjd->ntj',loggamma(aW[None,:,None,:] + aAlpha[:,None,:,:])) # (nd)+(tjd)->(tnjd)
    out -= np.einsum('tjd->tj', loggamma(aAlpha))[None,:,:]
    return

def pt_logd_cumdircategorical_mx_ma_inplace_unstable_old(out, aW, aAlpha, sphere_mat):
    """
    inputs:
        aW:         (n,d)
        aAlpha:     (t,j,d)
        sphere_mat: (c,d)
    output:
        logd:       (n,t,j)
    """
    if aW.shape[1] == 0:
        return
    sa = np.einsum('tjd,cd->tjc', aAlpha, sphere_mat)
    ga = loggamma(aAlpha)
    ga1 = loggamma(aAlpha + 1)
    out += np.einsum('tjc->tj', loggamma(sa))[None,:,:]
    out -= np.einsum('tjc->tj', loggamma(sa + 1))[None,:,:]
    out += np.einsum('tjd,nd->ntj', ga1, aW)
    out += np.einsum('tjd,nd->ntj', ga, ~aW)
    out -= np.einsum('tjd->tj', ga)[None,:,:]
    return

def pt_logd_cumdircategorical_mx_ma(aW, aAlpha, sphere_mat):
    """
    inputs:
        aW:         (n,d)
        aAlpha:     (t,j,d)
        sphere_mat: (c,d)
    output:
        logd:       (n,t,j)
    """
    n = aW.shape[0]; t,j,d = aAlpha.shape
    out = np.zeros((n,t,j))
    sa = np.einsum('tjd,cd->tjc', aAlpha, sphere_mat)
    ga = loggamma(aAlpha)
    ga1 = loggamma(aAlpha + 1)
    out += np.einsum('tjc->tj', loggamma(sa))[None,:,:]
    out -= np.einsum('tjc->tj', loggamma(sa + 1))[None,:,:]
    out += np.einsum('tjd,nd->ntj', ga1, aW)
    out += np.einsum('tjd,nd->ntj', ga, ~aW)
    out -= np.einsum('tjd->tj', ga)[None,:,:]
    return out

def pt_logd_cumdircategorical_mx_ma_inplace_unstable(
        out         : np.ndarray, 
        aC          : np.ndarray, 
        aAlpha      : np.ndarray, 
        sphere_mat  : np.ndarray,
        ):
    """
    inputs:
        aC:         (n,d)
        aAlpha:     (t,j,d)
        sphere_mat: (c,d)
    outputs:
        logd:       (n,t,j)
    """
    if aC.shape[1] == 0:
        return
    lsa = np.log(aAlpha @ sphere_mat.T).sum(axis = -1) # (t,j)
    lca = np.einsum('nd,tjd->ntj', aC, np.log(aAlpha))
    out += lca
    out -= lsa
    return

def pt_logd_cumdircategorical_mx_ma(
        aC          : np.ndarray, 
        aAlpha      : np.ndarray, 
        sphere_mat  : np.ndarray,
        ):
    """
    inputs:
        aC:         (n,d)
        aAlpha:     (t,j,d)
        sphere_mat: (c,d)
    output:
        logd:       (n,t,j)
    """
    n = aC.shape[0]; t,j,d = aAlpha.shape
    out = np.zeros((n,t,j))
    lsa = np.log(aAlpha @ sphere_mat.T).sum(axis = -1) # (t,j)
    lca = np.einsum('nd,tjd->ntj', aC, np.log(aAlpha))
    out += lca
    out -= lsa
    return out

def pt_logd_cumdirmultinom_mx_ma_inplace_unstable(out, aW, aAlpha, sphere_mat):
    sa = np.einsum('tjd,cd->tjc', aAlpha, sphere_mat)
    sw = np.einsum('nd,cd->nc', aW, sphere_mat)
    out += np.einsum('tjc->tj', loggamma(sa))[None,:,:]
    out += np.einsum('nc->n', loggamma(sw + 1))[:,None,None]
    out -= np.einsum('tnjc->ntj', loggamma(sw[None,:,None,:] + sa[:,None,:,:])) # (n,c)+(tjc)->(tnjc)
    out += np.einsum('tnjd->ntj',loggamma(aW[None,:,None,:] + aAlpha[:,None,:,:])) # (nd)+(tjd)->(tnjd)
    out -= np.einsum('tjd->tj', loggamma(aAlpha))[None,:,:]
    out -= np.einsum('nd->n', loggamma(aW + 1))[:,None,None]
    return

def logd_cumdirmultinom_paired(aW, aAlpha, sphere_mat):
    sa = np.einsum('nd,cd->nc', aAlpha, sphere_mat)
    sw = np.einsum('nd,cd->nc', aW, sphere_mat)
    logd = np.zeros(aW.shape[0])
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        logd += np.einsum('nc->n', loggamma(sa))
        logd += np.einsum('nc->n', loggamma(sw + 1))
        logd -= np.einsum('nc->n', loggamma(sw + sa))
        logd += np.einsum('nd->n', loggamma(aW + aAlpha))
        logd -= np.einsum('nd->n', loggamma(aAlpha))
        logd -= np.einsum('nd->n', loggamma(aW + 1))
    np.nan_to_num(logd, False, -np.inf)
    return logd

def pt_logd_cumdirmultinom_paired_yt(aW, aAlpha, sphere_mat):
    """
    returns log-likelihood per temperature
    inputs:
        aW         : (n,d)
        aAlpha     : (t,n,d)
        sphere_mat : (c,d)
    outputs:
        logd       : (t,n)
    """
    if aAlpha.shape[-1] == 0:
        return np.zeros(aAlpha.shape[:-1])
    sa = np.einsum('tnd,cd->tnc', aAlpha, sphere_mat)
    sw = np.einsum('nd,cd->nc', aW.astype(int), sphere_mat)
    logd = np.zeros((aAlpha.shape[0], aW.shape[0]))
    # with np.errstate(divide = 'ignore', invalid = 'ignore'):
    with nullcontext():
        logd += np.einsum('tnc->tn', loggamma(sa))
        logd += np.einsum('nc->n', loggamma(sw + 1))[None,:]
        logd -= np.einsum('tnc->tn', loggamma(sw[None,:,:] + sa))
        logd += np.einsum('tnd->tn', loggamma(aW[None,:,:] + aAlpha))
        logd -= np.einsum('tnd->tn', loggamma(aAlpha))
        logd -= np.einsum('nd->n', loggamma(aW + 1))[None,:]
    # np.nan_to_num(logd, False, -np.inf)
    return logd

## Functions relating to loggamma density

def logd_loggamma_sx(x, a, b):
    logd = 0.
    logd += a * log(b)
    logd -= loggamma(a)
    logd += a * x
    logd -= b * exp(x)
    return logd

def logd_loggamma_mx_st(x, a, b):
    logd = np.zeros(x.shape[0])
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        logd += a * np.log(b)
        logd -= loggamma(a)
        logd += a * x
        logd -= b * np.exp(x)
    return logd

def logd_loggamma_paired(x, a, b):
    """
    Log-density of log-gamma distribution at multiple temperatures
    inputs:
        x   : (m,d)
        a   : (m,d)
        b   : (m,d)
    output:
        out : (m)
    """
    logd = np.empty((x.shape[0]))
    with np.errstate(divide = 'ignore', invalid = 'ignore', over = 'ignore'):
        logd += np.einsum('md,md->m', a, np.log(b))
        logd -= np.einsum('md->m', loggamma(a))
        logd += np.einsum('md,md->m', a, x)
        logd -= np.einsum('md,md->m', b, np.exp(x))
    np.nan_to_num(logd, False, -np.inf)
    return logd

def pt_logd_loggamma_mx_st(x, a, b):
    """
    Log-density of log-gamma distribution at multiple temperatures
    inputs:
        x : (t,j,d)
        a : (t,d)
        b : (t,d)
    output:
        out : (t,j)
    """
    logd = np.empty((x.shape[0], x.shape[1]))
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        logd += np.einsum('td,td->t', a, np.log(b))[:,None]
        logd -= np.einsum('td->t', loggamma(a))[:,None]
        logd += np.einsum('td,tjd->tj', a, x)
        logd -= np.einsum('td,tjd->tj', b, np.exp(x))
    np.nan_to_num(logd, False, -np.inf)
    return logd

def pt_logpost_loggammagamma(logalpha, n, sy, sly, a, b, c, d):
    """
    logalpha    : (t, d), log-shape parameter (variable) in gamma-gamma model
    n,          : (t), number of observations
    sy, sly     : (t, d), (number of obs, summed y, summed log-y)
    a, b        : (1), gamma prior parameters for shape
    c, d        : (1), gamma prior parameters for rate
    """
    alpha = np.exp(logalpha)
    ld = np.zeros(logalpha.shape)
    ld += (alpha - 1) * sly
    ld += a * logalpha
    ld -= b * alpha
    ld += loggamma(n[:,None] * alpha + c)
    ld -= (n[:,None] * alpha + c) * np.log(sy + d)
    return ld

def pt_logpost_lognormalgamma(logalpha, n, sy, sly, mu, sigma, xi, tau):
    """
    Log-posterior for shape parameter of gamma distribuion, wih rate
    integrated out.
    ---
    logalpha : (t, j, d)
    n        : (t, j)
    sy       : (t, j, d) : sum of y
    sly      : (t, j, d) : sum of log(y)
    mu       : (t, d)    : mean of log-normal prior for shape
    sig      : (t, d)    : scale of log-normal prior for shape
    xi       : (t, d)    : shape of gamma prior for rate
    tau      : (t, d)    : rate of gamma prior for rate
    """
    T, J, D = logalpha.shape
    alpha = np.exp(logalpha)
    out = np.zeros((T,J,D))
    out += (alpha - 1) * sly
    out -= n[:,:,None] * loggamma(alpha)
    out -= 0.5 * ((logalpha - mu[:,None]) / sigma[:,None])**2
    out += loggamma(n[:,:,None] * alpha + xi[:,None])
    out -= (n[:,None] * alpha + xi[:,None]) * np.log(sy + tau[:,None])
    return out

## functions relating to inverse gamma density
def logd_invgamma_my(aY, alpha, beta):
    """
    log-density of inverse gamma distribution
    ---
    Args:
        aY : (t,d)
        aAlpha : scalar
        aBeta  : scalar
    Returns:
        out : (t,d)
    """
    out = np.zeros(aY.shape)
    out += alpha * np.log(beta)
    out -= gammaln(alpha)
    out -= (alpha + 1) * np.log(aY)
    out -= aY * beta
    return out

## Functions relating to Pareto Density

def pt_logd_pareto_paired_yt(vR, aAlpha):
    """
    inputs:
        out : (t,n)
        vR  : (n)
        aAlpha : (t,n)
    """
    return -(aAlpha + 1) * np.log(vR[None])

def pt_logd_pareto_mx_ma_inplace_unstable(out, vR, aAlpha):
    """
    inputs:
        out:    (n,t,j)
        vR:     (n)
        aAlpha: (t,j)
    """
    if vR.shape[-1] == 0:
        return
    out -= (aAlpha + 1)[None] * np.log(vR[:,None,None])
    return

def pt_logd_pareto_mx_ma(vR, aAlpha):
    """
    inputs:
        vR:     (n)
        aAlpha: (t,j)
    outputs:
        out:    (n,t,j)
    """
    n = vR.shape[0]; t,j = aAlpha.shape
    out = np.zeros((n,t,j))
    out -= (aAlpha + 1)[None] * np.log(vR[:,None,None])
    return out

## Functions related to sampling for parameters from posterior, assuming
## a projected gamma likelihood.

def log_post_log_alpha_1(log_alpha_1, y_1, prior):
    """ Log posterior for log-alpha_1 assuming a gamma distribution,
    with beta assumed to be 1. """
    alpha_1 = exp(log_alpha_1)
    n_1     = y_1.shape[0]
    lp = (
        + prior.a * log_alpha_1
        - prior.b * alpha_1
        + (alpha_1 - 1) * np.log(y_1).sum()
        - n_1 * loggamma(alpha_1)
        )
    return lp

def sample_alpha_1_mh(curr_alpha_1, y_1, prior, proposal_sd = 0.3):
    """ Sampling function for shape parameter, with gamma likelihood and
    gamma prior.  Assumes rate parameter = 1.  uses Metropolis Hastings
    algorithm with random walk for sampling. """
    if len(y_1) < 1:
        return gamma.rvs(prior.a, scale = 1./prior.b)

    curr_log_alpha_1 = log(curr_alpha_1)
    prop_log_alpha_1 = curr_log_alpha_1 + normal.rvs(scale = proposal_sd)

    curr_lp = log_post_log_alpha_1(curr_log_alpha_1, y_1, prior)
    prop_lp = log_post_log_alpha_1(prop_log_alpha_1, y_1, prior)

    if log(uniform.rvs()) < prop_lp - curr_lp:
        return exp(prop_log_alpha_1)
    else:
        return curr_alpha_1

def log_post_log_alpha_k(log_alpha, y, prior_a, prior_b):
    """ Log posterior for log-alpha assuming a gamma distribution,
    beta integrated out of the posterior. """
    alpha = exp(log_alpha)
    n  = y.shape[0]
    lp = (
        + (alpha - 1) * np.log(y).sum()
        - n * loggamma(alpha)
        + prior_a.a * log_alpha
        - prior_a.b * alpha
        + loggamma(n * alpha + prior_b.a)
        - (n * alpha + prior_b.a) * log(y.sum() + prior_b.b)
        )
    return lp

def sample_alpha_k_mh(curr_alpha_k, y_k, prior_a, prior_b, proposal_sd = 0.3):
    """ Sampling Function for shape parameter, with Gamma likelihood and Gamma
    prior, with rate (with gamma prior) integrated out. """
    if len(y_k) <= 1:
        return gamma.rvs(prior_a.a, scale = 1./prior_a.b)

    curr_log_alpha_k = log(curr_alpha_k)
    prop_log_alpha_k = curr_log_alpha_k + normal.rvs(scale = proposal_sd)

    curr_lp = log_post_log_alpha_k(curr_log_alpha_k, y_k, prior_a, prior_b)
    prop_lp = log_post_log_alpha_k(prop_log_alpha_k, y_k, prior_a, prior_b)

    if log(uniform.rvs()) < prop_lp - curr_lp:
        return exp(prop_log_alpha_k)
    else:
        return curr_alpha_k

def sample_beta_fc(alpha, y, prior):
    aa = len(y) * alpha + prior.a
    bb = sum(y) + prior.b
    return gamma.rvs(aa, scale = 1. / bb)

## Test Densities
def lpv_inner(g, alpha, l):
    return np.prod(gammainc(np.delete(alpha, l), g)) * g**(alpha[l] - 1) * np.exp(-g)

def log_prob_max_of_gammas(alpha, l):
    """
    alpha = gamma shape variables
    l     = dimension of vector to find max
    """
    lpv = - loggamma(alpha).sum()
    lpv += np.log(quad(lambda g: lpv_inner(g, alpha, l), 0, np.inf)[0])
    return lpv 

def logd_projgamma_linf(v, alpha):
    l = np.argmax(v)
    ld = (
        + log_prob_max_of_gammas(alpha, l)
        + ((alpha - 1) * np.log(v)).sum()
        - loggamma(alpha).sum()
        + loggamma(alpha.sum())
        - alpha.sum() * np.log(v.sum())
        )
    return ld

def logd_gamma(Y, alpha, beta):
    """
    gamma logdensity
    ----
    args:
        Y: (...)
        alpha: scalar
        beta:  scalar
    return:
        logd: Y.shape
    """
    out = np.zeros(Y.shape)
    out += alpha * log(beta)
    out -= loggamma(alpha)
    out += (alpha - 1) * np.log(Y)
    out -= beta * Y
    return out

if __name__ == '__main__':
    pass

# EOF
