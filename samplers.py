"""
samplers.py
--------------
Proto-classes for MCMC samplers.

both assume the existence of:
    obj.initialize_sampler(ns)
    obj.iter_sample()
    obj.curr_iter

DirichletProcessSampler assumes the existence of:
    obj.samples.delta
    obj.curr_delta
"""
import time
import numpy as np
np.seterr(divide = 'raise', invalid = 'raise')
from numpy.random import beta, uniform, gamma
from scipy.special import loggamma, betaln, softmax
from collections import namedtuple
import math
import warnings
# rng = np.random.default_rng(seed = inr(time.time())

# set_num_threads(4)

EPS = np.finfo(float).eps
MAX = np.finfo(float).max

GEMPrior = namedtuple('GEMPrior', 'discount concentration')

def bincount2D_vectorized(arr, m):
    """
    code from stackoverflow:
        https://stackoverflow.com/questions/46256279    
    Args:
        arr : (np.ndarray(int)) -- matrix of cluster assignments by temperature (t x n)
        m   : (int)             -- maximum number of clusters
    Returns:
        (np.ndarray(int)): matrix of cluster counts by temperature (t x J)
    """
    arr_offs = arr + np.arange(arr.shape[0])[:,None] * m
    return np.bincount(arr_offs.ravel(), minlength=arr.shape[0] * m).reshape(-1, m)

class BaseSampler(object):
    print_string_during = '\rSampling {:.1%} Completed in {}'
    print_string_after = '\rSampling 100% Completed in {}'

    @property
    def time_elapsed_numeric(self):
        return time.time() - self.start_time

    @property
    def time_elapsed(self):
        """ returns current time elapsed since sampling start in human readable format """
        elapsed = self.time_elapsed_numeric
        if elapsed < 60:
            return '{:.0f} Seconds'.format(elapsed)
        elif elapsed < 3600: 
            return '{:.2f} Minutes'.format(elapsed / 60)
        else:
            return '{:.2f} Hours'.format(elapsed / 3600)
        pass

    def sample(self, ns, verbose = False):
        """ Run the Sampler """
        self.initialize_sampler(ns)
        self.start_time = time.time()
        
        if verbose:
            print('\rSampling 0% Completed', end = '')

        while (self.curr_iter < ns):
            if (self.curr_iter % 100) == 0:
                ps = self.print_string_during.format(self.curr_iter / ns, self.time_elapsed)
                if verbose:
                    print(ps.ljust(80), end = '')
            self.iter_sample()
        
        ps = self.print_string_after.format(self.time_elapsed)
        if verbose:
            print(ps)
        return

class DirichletProcessSampler(BaseSampler):
    print_string_during = '\rSampling {:.1%} Completed in {}, {} Clusters'
    print_string_after  = '\rSampling 100% Completed in {}, {} Clusters Avg.'
    
    @property
    def curr_cluster_count(self):
        """ Returns current cluster count """
        return self.curr_delta.max() + 1
    
    def average_cluster_count(self, ns):
        acc = self.samples.delta[(ns//2):].max(axis = 1).mean() + 1
        return '{:.2f}'.format(acc)

    def sample(self, ns, verbose = False):
        """ Run the sampler """
        self.initialize_sampler(ns)
        self.start_time = time.time()
        
        if verbose:
            print('\rSampling 0% Completed', end = '')
        
        while (self.curr_iter < ns):
            if (self.curr_iter % 100) == 0:
                ps = self.print_string_during.format(
                    self.curr_iter / ns, self.time_elapsed, self.curr_cluster_count,
                    )
                if verbose:
                    print(ps.ljust(80), end = '')
            self.iter_sample()
        
        ps = self.print_string_after.format(self.time_elapsed, self.average_cluster_count(ns))
        if verbose:
            print(ps)
        return

class StickBreakingSampler(DirichletProcessSampler):
    @property
    def curr_cluster_count(self):
        return (np.bincount(self.curr_delta) > 0).sum()
    
    def average_cluster_count(self, ns):
        try:
            cc = bincount2D_vectorized(
                self.samples.delta[(ns//2):], 
                self.samples.delta.max() + 1,
                )
        except TypeError:
            cc = bincount2D_vectorized(
                np.stack(self.samples.delta),
                np.stack(self.samples.delta).max() + 1,
                )
        return (cc > 0).sum(axis = 1).mean()

def dp_sample_cluster_crp8(delta, log_likelihood, prob, eta):
    '''
    Args:
        delta          : (N)      : current cluster assignment
        log_likelihood : (N x J)  : log-likelihood of obs n under cluster j
        prob           : (N)      : vector of random uniforms
        eta            : (scalar) : concentration parameter
    Note:
        Modifies delta in place.
    '''
    N, J = log_likelihood.shape
    curr_cluster_state = np.bincount(delta, minlength = J)
    cand_cluster_state = (curr_cluster_state == 0)
    scratch = np.empty(curr_cluster_state.shape)
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        for n in range(N):
            curr_cluster_state[delta[n]] -= 1
            scratch[:] = curr_cluster_state
            scratch += cand_cluster_state * (eta / (cand_cluster_state.sum() + EPS))
            np.log(scratch, out = scratch)
            scratch[np.isnan(scratch)] = -np.inf
            scratch += log_likelihood[n]
            scratch -= scratch.max()
            np.exp(scratch, out = scratch)
            np.cumsum(scratch, out = scratch)
            scratch /= scratch[-1]
            delta[n] = (prob[n] > scratch).sum()
            curr_cluster_state[delta[n]] += 1
            cand_cluster_state[delta[n]] = False
    return

def pt_dp_sample_cluster_crp8(delta, log_likelihood, prob, eta):
    """
    Args:
        delta          : (T x N)
        log_likelihood : (N x T x J)
        prob ([type])  : (N x T)
        eta ([type])   : (T)
    """
    N, T, J = log_likelihood.shape
    curr_cluster_state = bincount2D_vectorized(delta, J)
    cand_cluster_state = (curr_cluster_state == 0)
    scratch = np.empty(curr_cluster_state.shape)
    temps = np.arange(T)
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        for n in range(N):
            curr_cluster_state[temps, delta.T[n]] -= 1
            scratch[:] = curr_cluster_state
            scratch += cand_cluster_state * (eta / (cand_cluster_state.sum(axis = 1) + EPS))[:,None]
            np.log(scratch, out = scratch)
            scratch[np.isnan(scratch)] = -np.inf
            scratch += log_likelihood[n]
            scratch -= scratch.max(axis = 1)[:,None]
            np.exp(scratch, out = scratch)
            np.cumsum(scratch, axis = 1, out = scratch)
            scratch /= scratch[:,-1][:,None]
            delta.T[n] = (prob[n][:,None] > scratch).sum(axis = 1)
            curr_cluster_state[temps, delta.T[n]] += 1
            cand_cluster_state[temps, delta.T[n]] = False
    return

def dp_sample_chi_bgsb(delta, eta, J):
    '''
    args: 
        delta : (N)    : current cluster assignment
        eta   : scalar : concentration parameter
        J     : scalar : Blocked-Gibbs Stick-breaking Truncation Point
    Out: 
        chi   : (J)
    '''
    clustcount = np.bincount(delta, minlength = J)
    chi = beta(
        a = 1 + clustcount,
        b = eta + clustcount[::-1].cumsum()[::-1] - clustcount,
        )
    return chi

def pt_dp_sample_chi_bgsb(delta, eta, J):
    """
    Args: 
        delta : (T, N)
        eta   : (T)
        J     : Scalar (Blocked-Gibbs Stick-Breaking Truncation Point)
    Out:
        chi   : (T, J)
    """
    clustcount = bincount2D_vectorized(delta, J)
    # chi = beta(
    #     a = 1 + clustcount,
    #     b = eta[:,None] + clustcount[:,::-1].cumsum(axis = 1)[:,::-1] - clustcount,
    #     )
    # return chi
    A = gamma(1 + clustcount)
    B = gamma(eta[:,None] + clustcount[:,::-1].cumsum(axis = 1)[:,::-1] - clustcount)
    return np.exp(np.log(A) - np.log(A + B))
    
def dp_sample_concentration_bgsb(chi, a, b):
    """
    Gibbs Sampler for Concentration Parameter under
        Blocked-Gibbs Sticking-Breaking Representation of DP
    Args:
        chi : (J)
        a   : scalar
        b   : scalar
    Out:
        eta : scalar
    """
    eta = gamma(
        shape = a + chi.shape[0] - 1,
        scale = 1 / (b - np.log(1 - chi[:-1]).sum())
        )
    return eta

def pt_dp_sample_concentration_bgsb(chi, a, b):
    """
    Gibbs Sampler for Concentration Parameter under
        Blocked-Gibbs Sticking-Breaking Representation of DP
    Args:
        chi : (T, J)
        a   : scalar
        b   : scalar
    Out:
        eta : (T)
    """
    _shape = a + chi.shape[1] - 1
    _scale = 1 / (b - np.log(np.maximum(1 - chi[:,:-1], 1e-9)).sum(axis = 1))
    return gamma(shape = _shape, scale = _scale)

def dp_sample_cluster_bgsb(chi, log_likelihood):
    """
    Args:
        chi            : (J)     : Random Weights (betas)  (should be J - 1; fixed here)
        log_likelihood : (N x J) : log-likelihood of obs n under cluster j
        eta ([type])   : Scalar
    """
    N, J = log_likelihood.shape
    scratch = np.zeros((N,J))
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        scratch += np.log(np.hstack(chi[:,-1],(0,))) # scalar
        scratch += np.hstack(((0,),np.log(1 - chi[:-1]).cumsum())) # (J)
        scratch += log_likelihood # (N, J)
        scratch[np.isnan(scratch)] = -np.inf
        scratch -= scratch.max(axis = 1)[:,None]
        np.exp(scratch, out = scratch)
        np.cumsum(scratch, axis = 1, out = scratch)
        scratch /= scratch[:-1][:,None]
    delta = (uniform(size = (N))[:,None] > scratch).sum(axis = 1)
    return delta

def py_sample_cluster_bgsb_fixed(chi, log_likelihood):
    """
    Args:
        chi            : (J - 1  : Random Weights (betas)
        log_likelihood : (N x J) : log-likelihood of obs n under cluster j
        eta ([type])   : Scalar
    """
    N, J = log_likelihood.shape
    scratch = np.zeros((N,J))
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        scratch[:,:-1] += np.log(chi)
        scratch[:,1: ] += np.cumsum(np.log(1 - chi), axis = -1)
        scratch += log_likelihood # (N, J)
        scratch[np.isnan(scratch)] = -np.inf
        scratch -= scratch.max(axis = 1)[:,None]
        np.exp(scratch, out = scratch)
        np.cumsum(scratch, axis = 1, out = scratch)
        scratch /= scratch[:,-1][:,None]
    delta = (uniform(size = (N))[:,None] > scratch).sum(axis = 1)
    return delta

def py_sample_cluster_bgsb(chi, log_likelihood):
    return dp_sample_cluster_bgsb(chi, log_likelihood)

def pt_dp_sample_cluster_bgsb(chi, log_likelihood):
    """
    Args:
        chi            : (T, J)
        log_likelihood : (N x T x J)
        eta ([type])   : (T)
    """
    N, T, J = log_likelihood.shape
    scratch = np.zeros((N, T, J))
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        scratch += np.hstack(
            (np.log(chi[:,:-1]), np.zeros((T,1))),
            )[None] # Log Prior (part 1)
        scratch += np.hstack(        # Log Prior (part 2)
            (np.zeros((T,1)), np.log(1 - chi[:,:-1]).cumsum(axis = 1)),
            )[None]
        scratch += log_likelihood # log likelihood
        scratch[np.isnan(scratch)] = -np.inf
        scratch -= scratch.max(axis = 2)[:,:,None]
        np.exp(scratch, out = scratch)
        np.cumsum(scratch, axis = 2, out = scratch)
        scratch /= scratch[:,:,-1][:,:,None]
    # delta = (uniform(size = (N,T))[:,:,None] > scratch).sum(axis = 2).T
    delta = (
        (uniform(size = (N,T))[:,:,None] > scratch) @ np.ones(J, dtype = int)
        ).T
    return delta

def pt_py_sample_cluster_bgsb_fixed(chi, log_likelihood):
    """
    Args:
        chi            : (T, J - 1)
        log_likelihood : (N x T x J)
    """
    N, T, J = log_likelihood.shape
    scratch = np.zeros((N, T, J))
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        scratch[..., :-1] += np.log(chi)[None]
        scratch[..., 1: ] += np.cumsum(np.log(1 - chi), axis = -1)[None]
        scratch += log_likelihood 
    scratch[np.isnan(scratch)] = -np.inf
    probs = softmax(scratch, axis = -1) # handles shifting internally to avoid overflows.
    np.cumsum(probs, axis = -1, out = probs)
    delta = (
        (uniform(size = (N,T,1)) > probs) @ np.ones(J, dtype = int)
        ).T
    return delta

def pt_dp_sample_cluster(delta, log_likelihood, prob, eta):
    return pt_dp_sample_cluster_crp8(delta, log_likelihood, prob, eta)

def pt_py_sample_chi_bgsb(delta, disc, conc, trunc):
    """
    Args:
        delta : (T x N)
        disc  : scalar (Pitman Yor discount parameter)
        conc  : scalar (Pitman Yor concentration parameter)
        trunc : Scalar (Blocked Gibbs Stick-Breaking Truncation Point)
    Out:
        chi   : (T, trunc)
    """
    clustcount = bincount2D_vectorized(delta, trunc)
    shape1 = 1 - disc + clustcount
    shape2 = (
        + conc
        + (np.arange(trunc) + 1)[None] * disc
        + np.flip(np.flip(clustcount, -1).cumsum(axis = -1), -1) - clustcount
        )
    chi = beta(a = shape1, b = shape2)
    return chi

def pt_py_sample_chi_bgsb_fixed(delta, disc, conc, trunc):
    """
    Args:
        delta : (T x N)
        disc  : scalar (Pitman Yor discount parameter)
        conc  : scalar (Pitman Yor concentration parameter)
        trunc : Scalar (Blocked Gibbs Stick-Breaking Truncation Point)
    Out:
        chi   : (T, trunc - 1)
    """
    clustcount = bincount2D_vectorized(delta, trunc)
    shape1 = 1 - disc + clustcount
    shape2 = (
        + conc
        + (np.arange(trunc) + 1)[None,:] * disc
        + np.flip(np.flip(clustcount, -1).cumsum(axis = -1), -1) - clustcount
        )
    chi = beta(a = shape1[:,:-1], b = shape2[:,:-1])
    return chi

def py_sample_chi_bgsb_fixed(delta, disc, conc, trunc):
    clustcount = np.bincount(delta, minlength = trunc)
    shape1 = 1 + clustcount - disc
    shape2 = (
        + conc
        + clustcount[::-1].cumsum()[::-1] - clustcount
        + (np.arange(trunc) + 1) * disc
    )
    chi = beta(a = shape1[:-1], b = shape2[:-1])
    return chi

def pt_py_sample_cluster_bgsb(chi, log_likelihood):
    return pt_dp_sample_cluster_bgsb(chi, log_likelihood)

def pt_logd_gem_mx_st(chi, conc, disc):
    """ 
    Log-density for Griffith, Engen, & McCloskey distribution.
    Args:
        chi  : (T, J)
        conc : scalar
        disc : scalar
    """
    if type(conc) is not np.ndarray:
        conc = np.array([conc])
    k = (np.arange(chi.shape[1] - 1) + 1).reshape(1,-1)
    a = (1 - disc) * np.ones(k.shape)
    b = conc[:,None] + k * disc
    ld = np.zeros(chi.shape[0])
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        ld += ((a - 1) * np.log(chi[:,:-1])).sum(axis = 1)
        ld += ((b - 1) * np.log(1 - chi[:,:-1])).sum(axis = 1)
        ld -= betaln(a,b).sum(axis = 1)
    return ld

def pt_logd_gem_mx_st_fixed(chi, disc, conc):
    """ 
    Log-density for Griffith, Engen, & McCloskey distribution.
    Args:
        chi  : (T, J-1)
        disc : scalar
        conc : scalar
    """
    if type(conc) is not np.ndarray:
        conc = np.array([conc])
    k = (np.arange(chi.shape[1]) + 1).reshape(1,-1)
    a = (1 - disc) * np.ones(k.shape)
    b = conc[:,None] + k * disc
    ld = np.zeros(chi.shape[0])
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        ld += ((a - 1) * np.log(chi)).sum(axis = 1)
        ld += ((b - 1) * np.log(1 - chi)).sum(axis = 1)
        ld -= betaln(a,b).sum(axis = 1)
    return ld

class ParallelTemperingCRPSampler(DirichletProcessSampler):
    @property
    def curr_cluster_count(self):
        return self.curr_delta[0].max() + 1
    def average_cluster_count(self, ns):
        acc = self.samples.delta[(ns//2):,0].max(axis = 1).mean() + 1
        return '{:.2f}'.format(acc)

class ParallelTemperingStickBreakingSampler(DirichletProcessSampler):
    @property
    def curr_cluster_count(self):
        return (np.bincount(self.curr_delta[0]) > 0).sum()
    def average_cluster_count(self, ns):
        cc = bincount2D_vectorized(
            self.samples.delta[(ns//2):,0], 
            self.samples.delta[:,0].max() + 1,
            )
        return '{:.2f}'.format((cc > 0).sum(axis = 1).mean()) 

class Stepsize(object):
    """ Stepsize calculator for adaptive Metropolis (univariate) """
    shape = None
    init_stepsize = None
    curr_log_stepsize = None
    nIter = None
    update_interval = None
    succeed = None
    
    @property
    def step(self):
        return self.curr_stepsize

    def update(self, succeed, fail):
        self.succeed += succeed
        self.nIter += 1

        if self.nIter == self.update_interval:
            prob = self.succeed / self.update_interval
            self.succeed[:] = 0
            self.curr_stepsize[prob > 0.30] *= 1.2
            self.curr_stepsize[prob < 0.20] *= 0.8
        return

    def __init__(self, shape, init_stepsize = 0.2, update_interval = 100,):
        self.shape = shape
        self.init_stepsize = init_stepsize
        self.curr_stepsize = np.ones(shape) * init_stepsize
        self.succeed = np.zeros(shape, dtype = int)
        self.nIter = 0
        self.update_interval = update_interval
        return
    
    pass 

if __name__ == '__main__':
    pass



# EOF
