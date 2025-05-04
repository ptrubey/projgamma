import numpy as np
import numpy.typing as npt
from numpy.random import choice, gamma, uniform, normal, lognormal
from collections import namedtuple, deque
import pandas as pd
import os
import pickle
from scipy.special import digamma
from io import BytesIO
# Custom Modules
from samplers import py_sample_chi_bgsb, py_sample_cluster_bgsb,               \
    StickBreakingSampler, stickbreak
from data import euclidean_to_hypercube, Data
from projgamma import GammaPrior, logd_projgamma_my_mt_inplace_unstable

np.seterr(divide = 'raise', over = 'raise', under = 'ignore', invalid = 'raise')

Prior = namedtuple('Prior','alpha beta')

class Samples(object):
    r     : deque[npt.NDArray[np.float64]] # radius (projected gamma)
    chi   : deque[npt.NDArray[np.float64]] # stick-breaking weights (unnormalized)
    delta : deque[npt.NDArray[np.int32]]   # cluster identifiers
    beta  : deque[npt.NDArray[np.float64]] # rate hyperparameter
    alpha : deque[npt.NDArray[np.float64]] # shape hyperparameter (inferred)
    zeta  : deque[npt.NDArray[np.float64]] # shape parameter (inferred)
    
    def __init__(
            self, 
            nkeep : int, 
            N : int,  # nDat
            S : int,  # nCol
            J : int,  # nClust
            ):
        self.r     = deque([], maxlen = nkeep)
        self.chi   = deque([], maxlen = nkeep)
        self.delta = deque([], maxlen = nkeep)
        self.beta  = deque([], maxlen = nkeep)
        
        self.r.append(lognormal(mean = 3, sigma = 1, size = N))
        self.chi.append(1 / np.arange(2, J + 1)[::-1]) # uniform probability
        self.delta.append(choice(J, N))
        self.beta.append(gamma(shape = 2, scale = 1 / 2, size = S))
        
        self.alpha = deque([], maxlen = nkeep)
        self.zeta  = deque([], maxlen = nkeep)
        return
    
    pass

def gradient_resgammagamma_ln(
        theta   : np.ndarray,  # np.stack((mu, tau))    # (2, j, d)
        lYs     : np.ndarray,  # sum of log(Y)          # (j, d)
        n       : np.ndarray,  # number of observations # (j)
        a       : np.ndarray,  # hierarchical shape     # (d) or float
        b       : np.ndarray,  # hierarchical rate      # (d) or float
        ns      : int = 20,
        ) -> npt.NDArray[np.float64]:
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

def gradient_gammagamma_ln(
        theta   : npt.NDArray[np.float64],  # np.stack((mu, tau))
        lYs     : npt.NDArray[np.float64],  # sum of log(Y)
        Ys      : npt.NDArray[np.float64],  # sum of Y
        n       : npt.NDArray[np.int32],  # number of observations
        a       : npt.NDArray[np.float64],  # hierarchical (shape) shape 
        b       : npt.NDArray[np.float64],  # hierarchical (shape) rate
        c       : npt.NDArray[np.float64],  # hierarchical (rate) shape
        d       : npt.NDArray[np.float64],  # hierarchical (rate) rate
        ns      : int = 10,
        ) -> npt.NDArray[np.float64]:
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

class Adam(object):
    # Adam Parameters
    eps    : float = 1e-8
    rate   : float
    decay1 : float
    decay2 : float
    iter   : int
    niter  : int

    # Adam Updateables 
    momentum : npt.NDArray[np.float64] # momentum
    sumofsqs : npt.NDArray[np.float64] # sum of squares of past gradients
    theta    : npt.NDArray[np.float64] # parameter set

    # Loss function
    dloss : function[npt.NDArray[np.float64]] = None # function of theta

    def update(self) -> None:
        self.iter += 1
        dloss = self.dloss()
        self.momentum[:] = (
            + self.decay1 * self.momentum
            + (1 - self.decay1) * dloss
            )
        self.sumofsqs[:] = (
            + self.decay2 * self.sumofsqs
            + (1 - self.decay2) * dloss * dloss
            )
        mhat = self.momentum / (1 - self.decay1**self.iter)
        shat = self.sumofsqs / (1 - self.decay2**self.iter)
        self.theta -= mhat * self.rate / (np.sqrt(shat) + self.eps)
        return
    
    def specify_dloss(
            self, 
            func : function[npt.NDArray[np.float64]],
            ) -> None:
        self.dloss = func
        return

    def initialization(
            self, 
            rate   : float, # Adam Learning Rate
            decay1 : float, # Adam Decay 1
            decay2 : float, # Adam Decay 2
            niter  : int,   # Number of Adam Iterations per sample
            ) -> None:
        self.decay1 = decay1
        self.decay2 = decay2
        self.rate = rate
        self.iter = 0
        self.niter = niter
        self.momentum = np.zeros(self.theta.shape)
        self.sumofsqs = np.zeros(self.theta.shape)
        return

    def optimize(self) -> None:
        for _ in range(self.niter):
            self.update()
        return
    
    def __init__(
            self, 
            theta  : npt.NDArray[np.float64], 
            rate   : float = 1e-3, 
            decay1 : float = 0.9, 
            decay2 : float = 0.999, 
            niter  : float = 10,
            ):
        self.theta = theta
        self.initialization(rate, decay1, decay2, niter)
        return

class VariationalParameters(object):
    zeta_mutau   : npt.NDArray[np.float64]
    zeta_adam    : Adam
    alpha_mutau  : npt.NDArray[np.float64]
    alpha_adam   : Adam

    def __init__(self, S : int, J : int, **kwargs):
        self.zeta_mutau = np.zeros((2, J, S)) # normal(size = (2, J, S))
        self.alpha_mutau = np.zeros((2, S))   # normal(size = (2, S))

        self.zeta_adam = Adam(self.zeta_mutau, **kwargs)
        self.alpha_adam = Adam(self.alpha_mutau, **kwargs)
        return
    pass

class Chain(StickBreakingSampler):
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
            mean = self.varparm.alpha_mutau[0], 
            sigma = np.exp(self.varparm.alpha_mutau[1])
            )
    @property
    def curr_beta(self) -> npt.NDArray[np.float64]:
        return self.samples.beta[-1]
    @property
    def curr_zeta(self) -> npt.NDArray[np.float64]:
        return lognormal(
            mean = self.varparm.zeta_mutau[0], 
            sigma = np.exp(self.varparm.zeta_mutau[1]),
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
        self.varparm.zeta_adam.update
        
        func = lambda: - gradient_resgammagamma_ln(
            self.varparm.zeta_mutau, lYs, n, alpha, beta, self.var_samp,
            )

        self.varparm.zeta_adam.specify_dloss(func)
        self.varparm.zeta_adam.optimize()
        
        self.samples.zeta.append(self.curr_zeta)
        return
    
    def update_alpha(
            self, 
            zeta  : npt.NDArray[np.float64], 
            delta : npt.NDArray[np.float64],
            ) -> None:
        active = np.where(np.bincount(delta, minlength = self.J) > 0)[0]
        n = np.array((active.shape[0],))
        lZs = np.log(zeta)[active].sum(axis = 0)
        Zs  = zeta[active].sum(axis = 0)

        func = lambda: - gradient_gammagamma_ln(
            self.varparm.alpha_mutau, 
            lZs, Zs, n,
            *self.priors.alpha,
            *self.priors.beta,
            ns = self.var_samp
            )
                
        self.varparm.alpha_adam.specify_dloss(func)
        self.varparm.alpha_adam.optimize()

        self.samples.alpha.append(self.curr_alpha)
        return

    def update_beta(
            self, 
            zeta  : npt.NDArray[np.float64], 
            alpha : npt.NDArray[np.float64], 
            delta : npt.NDArray[np.int32],
            ) -> None:
        active = np.where(np.bincount(delta, minlength = self.J) > 0)[0]
        n = active.shape[0]
        zs = zeta[active].sum(axis = 0)
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
        r[r < 1e-4] = 1e-4               # lower-bounding radius
        self.samples.r.append(r)
        return
    
    def update_chi(self, delta : npt.NDArray[np.int32]) -> None:
        chi = py_sample_chi_bgsb(
            delta, self.discount, self.concentration, self.J,
            )
        self.samples.chi.append(chi)
        return
    
    def update_delta(
            self, 
            zeta : npt.NDArray[np.float64], 
            chi  : npt.NDArray[np.float64],
            ) -> None:
        llk = np.zeros((self.N, self.J))
        logd_projgamma_my_mt_inplace_unstable(
            llk, self.data.Yp, zeta, np.ones(zeta.shape),
            )
        delta = py_sample_cluster_bgsb(chi, llk)
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
        self.samples = Samples(self.gibbs_samp, self.N, self.S, self.J)
        self.varparm = VariationalParameters(
            self.S, self.J, niter = self.var_iter,
            )
        self.curr_iter = 0
        pass

    def set_projection(self) -> None:
        self.data.Yp = (
            self.data.V.T / 
            (self.data.V ** self.p).sum(axis = 1)**(1/self.p)
            ).T
        return
    
    def write_to_disk(self, path) -> None:
        if type(path) is str:
            folder = os.path.split(path)[0]
            if not os.path.exists(folder):
                os.mkdir(folder)
            if os.path.exists(path):
                os.remove(path)
        out = {
            'zeta_mutau'  : self.varparm.zeta_mutau,
            'alpha_mutau' : self.varparm.alpha_mutau,
            'zetas'       : np.stack(self.samples.zeta),
            'alphas'      : np.stack(self.samples.alpha),
            'betas'       : np.stack(self.samples.beta),
            'rs'          : np.stack(self.samples.r),
            'deltas'      : np.stack(self.samples.delta),
            'chis'        : np.stack(self.samples.chi),
            'time'        : self.time_elapsed_numeric,
            'conc'        : self.concentration,
            'disc'        : self.discount,
            'data'        : self.data.to_dict(),
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
            concentration : float = 0.1, 
            discount      : float = 0.1,
            ):
        self.data = data
        assert len(self.data.cats) == 0
        self.N = self.data.nDat
        self.S = self.data.nCol
        self.J = max_clusters
        self.p = p
        self.concentration = concentration
        self.discount = discount
        self.var_samp = var_samp
        self.var_iter = var_iter
        self.gibbs_samp = gibbs_samples
        self.priors = Prior(GammaPrior(*prior_alpha), GammaPrior(*prior_beta))
        self.set_projection()
        return

class ResultSamples(Samples):
    r     : npt.NDArray[np.float64]
    chi   : npt.NDArray[np.float64]
    delta : npt.NDArray[np.int32]
    beta  : npt.NDArray[np.float64]
    alpha : npt.NDArray[np.float64]
    zeta  : npt.NDArray[np.float64]

    def __init__(self, dict):
        self.r     = dict['rs']
        self.chi   = dict['chis']
        self.delta = dict['deltas']
        self.beta  = dict['betas']
        self.alpha = dict['alphas']
        self.zeta  = dict['zetas']
        return

class Result(object):
    samples : ResultSamples
    discount : float
    concentration : float
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
    
    def generate_posterior_predictive_zetas(self, n_per_sample = 10) -> npt.NDArray[np.float64]:
        zetas = []
        probs = stickbreak(self.samples.chi)
        Sprob = np.cumsum(probs, axis = -1)
        unis  = uniform(size = (self.nSamp, n_per_sample))
        for s in range(self.nSamp):
            delta = np.searchsorted(Sprob[s], unis[s])
            zetas.append(self.samples.zeta[s][delta])
        zetas = np.vstack(zetas)
        return(zetas)
    
    def generate_posterior_predictive_gammas(self, n_per_sample = 10) -> npt.NDArray[np.float64]:
        zetas = self.generate_posterior_predictive_zetas(n_per_sample)
        return gamma(shape = zetas)

    def generate_posterior_predictive_hypercube(self, n_per_sample = 10) -> npt.NDArray[np.float64]:
        gammas = self.generate_posterior_predictive_gammas(n_per_sample)
        return euclidean_to_hypercube(gammas)

    def load_data(self, path):
        if type(path) is BytesIO:
            out = pickle.loads(path.getvalue())
        else:
            with open(path, 'rb') as file:
                out = pickle.load(file)
        self.samples = ResultSamples(out)
        self.concentration = out['conc']
        self.discount = out['disc']
        self.N = out['nDat']
        self.S = out['nCol']
        self.nSamp = self.samples.chi.shape[0]
        self.time_elapsed_numeric = out['time']
        return

    def __init__(self, path):
        self.load_data(path)
        return
    
if __name__ == '__main__':
    # pass
    raw = pd.read_csv('./datasets/ivt_updated_nov_mar.csv')
    # raw = pd.read_csv('./datasets/ivt_nov_mar.csv')
    data = Data.from_raw(
        raw, 
        x1ht_cols = np.arange(raw.shape[1]), 
        dcls = True, 
        xhquant = 0.95,
        )
    model = Chain(data, p = 10, gibbs_samples = 1000,)
    model.sample(5000, verbose = True)
    model.write_to_disk('./test/results.pkl')
    res = Result('./test/results.pkl')
    cond_zetas  = res.generate_conditional_posterior_predictive_zetas()
    cond_gammas = res.generate_conditional_posterior_predictive_gammas()
    zetas       = res.generate_posterior_predictive_zetas()
    gammas      = res.generate_posterior_predictive_gammas()
    raise

# EOF