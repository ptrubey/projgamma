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
import os
import pickle
import numpy as np
import numpy.typing as npt
np.seterr(divide = 'raise', invalid = 'raise')
from typing import Self, NamedTuple
from collections.abc import Callable
from numpy.random import beta, uniform, gamma
from scipy.special import betaln, softmax, log_softmax
from io import BytesIO

EPS = np.finfo(float).eps
MAX = np.finfo(float).max

def bincount2D_vectorized(arr : npt.NDArray[np.int32], m : int) -> npt.NDArray[np.int32]:
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
    print_string_before : str = '\rSampling 0% Completed'
    print_string_during : str = '\rSampling {:.1%} Completed in {}'
    print_string_after  : str = '\rSampling 100% Completed in {}'
    curr_iter : int
    start_time : float

    def initialize_sampler(self, ns : int) -> None:
        raise NotImplementedError('Replace me!')
    def iter_sample(self) -> None:
        raise NotImplementedError('Replace me!')
    
    @property
    def time_elapsed_numeric(self) -> float:
        return time.time() - self.start_time

    @property
    def time_elapsed(self) -> str:
        """ returns current time elapsed since sampling start in human readable format """
        elapsed = self.time_elapsed_numeric
        if elapsed < 60:
            return '{:.0f} Seconds'.format(elapsed)
        elif elapsed < 3600: 
            return '{:.2f} Minutes'.format(elapsed / 60)
        else:
            return '{:.2f} Hours'.format(elapsed / 3600)
        pass

    def sample(
            self, 
            ns : int, 
            verbose : bool = False,
            print_interval : int = 100,
            ) -> None:
        """ Run the Sampler """
        self.initialize_sampler(ns)
        self.start_time = time.time()
        
        if verbose:
            print(self.print_string_before, end = '')

        while (self.curr_iter < ns):
            if verbose:
                if (self.curr_iter % print_interval) == 0:
                    ps = self.print_string_during.format(
                        self.curr_iter / ns, self.time_elapsed,
                        )
                    print(ps.ljust(80), end = '')
            self.iter_sample()
        
        ps = self.print_string_after.format(self.time_elapsed)
        if verbose:
            print(ps)
        return

class SamplesBase(object):
    def to_dict(self, nBurn : int, nThin : int) -> dict:
        raise NotImplementedError('Replace me!')
    
    @classmethod
    def from_dict(cls, out : dict) -> Self:
        return cls(**out)
    
    @classmethod
    def from_meta(
            cls, 
            nSamp  : int, 
            nDat   : int, 
            nCol   : int, 
            nClust : int,
            ) -> Self:
        raise NotImplementedError('Replace me!')
    pass

class VariationalBase(object):
    def to_dict(self) -> dict:
        raise NotImplementedError('Replace me!')
    @classmethod
    def from_dict(cls, out) -> Self:
        raise NotImplementedError('Replace me!')
    @classmethod
    def from_meta(cls, **kwargs) -> Self:
        raise NotImplementedError('Replace me!')
    pass

class CRPSampler(BaseSampler):
    print_string_during : str = '\rSampling {:.1%} Completed in {}, {} Clusters'
    print_string_after  : str = '\rSampling 100% Completed in {}, {} Clusters Avg.'
    samples : SamplesBase

    @property
    def curr_delta(self) -> npt.NDArray[np.int32]:
        raise NotImplementedError('Replace me!')
    
    @property
    def curr_cluster_count(self) -> int:
        """ Returns current cluster count """
        return self.curr_delta.max() + 1
    
    def average_cluster_count(self, ns : int) -> float:
        acc = self.samples.delta[(ns//2):].max(axis = 1).mean() + 1
        return '{:.2f}'.format(acc)

    def sample(self, ns : int, verbose : bool = False) -> None:
        """ Run the sampler """
        self.initialize_sampler(ns)
        self.start_time = time.time()
        
        if verbose:
            print(self.print_string_before, end = '')
        
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

class StickBreakingSampler(CRPSampler):
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

def dp_sample_cluster_crp8(
        delta          : npt.NDArray[np.int32], 
        log_likelihood : npt.NDArray[np.float64], 
        prob           : npt.NDArray[np.float64], 
        eta            : float,
        ) -> None:
    '''
    Sample cluster assignment via Algorithm 8 of Neal (2000).
    ----
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

def pt_dp_sample_cluster_crp8(
        delta          : npt.NDArray[np.int32], 
        log_likelihood : npt.NDArray[np.float64], 
        prob           : npt.NDArray[np.float64], 
        eta            : npt.NDArray[np.float64],
        ) -> None:
    """
    Sample cluster assignment via Algorithm 8 of Neal (2000).
    ---
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

def dp_sample_chi_bgsb(
        delta : npt.NDArray[np.int32], 
        eta   : float, 
        J     : int,
        ) -> npt.NDArray[np.float64]:
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

def pt_dp_sample_chi_bgsb(
        delta : npt.NDArray[np.int32],
        eta   : npt.NDArray[np.float64], 
        J     : int,
        ) -> npt.NDArray[np.float64]:
    """
    Args: 
        delta : (T, N)
        eta   : (T)
        J     : Scalar (Blocked-Gibbs Stick-Breaking Truncation Point)
    Out:
        chi   : (T, J)
    """
    clustcount = bincount2D_vectorized(delta, J)
    A = gamma(1 + clustcount)
    B = gamma(eta[:,None] + clustcount[:,::-1].cumsum(axis = 1)[:,::-1] - clustcount)
    return np.exp(np.log(A) - np.log(A + B))
    
def dp_sample_concentration_bgsb(
        chi : npt.NDArray[np.float64], 
        a   : float, 
        b   : float,
        ) -> float:
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

def pt_dp_sample_concentration_bgsb(
        chi : npt.NDArray[np.float64], 
        a   : float, 
        b   : float,
        ) -> npt.NDArray[np.float64]:
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

def dp_sample_cluster_bgsb(
        chi            : npt.NDArray[np.float64], 
        log_likelihood : npt.NDArray[np.float64],
        ) -> npt.NDArray[np.int32]:
    """
    Args:
        chi            : (J - 1  : Random Weights (betas)
        log_likelihood : (N x J) : log-likelihood of obs n under cluster j
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

def py_sample_cluster_bgsb(
        chi            : npt.NDArray[np.float64], 
        log_likelihood : npt.NDArray[np.float64],
        ) -> npt.NDArray[np.int32]:
    return dp_sample_cluster_bgsb(chi, log_likelihood)

def pt_dp_sample_cluster_bgsb(
        chi            : npt.NDArray[np.float64],
        log_likelihood : npt.NDArray[np.float64],
        ) -> npt.NDArray[np.int32]:
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

def pt_py_sample_cluster_bgsb(
        chi            : npt.NDArray[np.float64], 
        log_likelihood : npt.NDArray[np.float64],
        ) -> npt.NDArray[np.int32]:
    return pt_dp_sample_cluster_bgsb(chi, log_likelihood)

def py_sample_chi_bgsb(
        delta : npt.NDArray[np.int32], 
        disc : float, 
        conc : float, 
        trunc : int,
        ) -> npt.NDArray[np.float64]:
    clustcount = np.bincount(delta, minlength = trunc)
    shape1 = 1 + clustcount - disc
    shape2 = (
        + conc
        + clustcount[::-1].cumsum()[::-1] - clustcount
        + (np.arange(trunc) + 1) * disc
    )
    chi = beta(a = shape1[:-1], b = shape2[:-1])
    return chi

def pt_py_sample_chi_bgsb(
        delta : npt.NDArray[np.int32], 
        disc  : float, 
        conc  : float, 
        trunc : int,
        ) -> npt.NDArray[np.float64]:
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

def pt_py_sample_cluster_bgsb(
        chi            : npt.NDArray[np.float64], 
        log_likelihood : npt.NDArray[np.float64],
        ) -> npt.NDArray[np.int32]:
    return pt_dp_sample_cluster_bgsb(chi, log_likelihood)

def pt_logd_gem_mx_st(
        chi  : npt.NDArray[np.float64], 
        conc : float, 
        disc : float,
        ) -> npt.NDArray[np.float64]:    
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

def stickbreak(nu : npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
        Stickbreaking cluster probability
        nu : (S x (J - 1))
    """
    with np.errstate(divide = 'ignore'):
        lognu = np.log(nu)
        log1mnu = np.log(1 - nu)
    out = np.zeros((nu.shape[0], nu.shape[1] + 1))
    out[...,:-1] += lognu
    out[..., 1:] += np.cumsum(log1mnu, axis = -1)
    return softmax(out, axis = -1)

def log_stickbreak(nu : npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """ Parallel Tempered stick-breaking function (log) """
    with np.errstate(divide = 'ignore'):
        lognu = np.log(nu)
        log1mnu = np.log(1 - nu)
    out = np.zeros((nu.shape[0], nu.shape[1] + 1))
    out[..., :-1] += np.log(nu)
    out[..., 1: ] += np.cumsum(np.log(1 - nu), axis = -1)
    return log_softmax(out, axis = -1)

class ParallelTemperingCRPSampler(CRPSampler):
    @property
    def curr_cluster_count(self) -> int:
        return self.curr_delta[0].max() + 1
    
    def average_cluster_count(self, ns) -> float:
        acc = self.samples.delta[(ns//2):,0].max(axis = 1).mean() + 1
        return '{:.2f}'.format(acc)

class ParallelTemperingStickBreakingSampler(StickBreakingSampler):
    @property
    def curr_cluster_count(self) -> int:
        return (np.bincount(self.curr_delta[0]) > 0).sum()
    
    def average_cluster_count(self, ns) -> float:
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

def write_to_disk(out : dict, path : str) -> None:
    assert type(path) is str
    folder = os.path.split(path)[0]
    if not os.path.exists(folder):
        os.mkdir(folder)
    if os.path.exists(path):
        os.remove(path)
    with open(path, 'wb') as file:
        pickle.dump(out, file)
    return

def read_from_disk(path : str) -> dict:
    with open(path, 'rb') as file:
        out = pickle.load(file)
    return out

def write_to_iostream(out : dict, iostream : BytesIO) -> None:
    assert type(iostream) is BytesIO
    iostream.write(pickle.dumps(out))
    return

def read_from_iostream(iostream : BytesIO) -> dict:
    out = pickle.loads(iostream.getvalue())
    return out

if __name__ == '__main__':
    pass



# EOF
