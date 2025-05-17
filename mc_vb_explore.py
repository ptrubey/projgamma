""" Comparison of MCMC and VB on a univariate gamma-gamma model. """
import numpy as np
from numpy.random import normal, uniform, gamma
from scipy.special import gammaln, digamma

from samplers import BaseSampler

class Samples(object):
    def __init__(self, ns):
        self.alpha = np.empty(ns + 1)
        self.beta  = np.empty(ns + 1)
        return
    pass

class UnivariateResGammaGammaMC(BaseSampler):
    @property
    def curr_alpha(self):
        return self.samples.alpha[self.curr_iter]
    
    def initialize_sampler(self, ns):
        self.samples = Samples(ns)
        self.samples.alpha[0] = 1.
        self.curr_iter = 0
        return

    def log_posterior_logalpha(self, sigma):
        return (
            + (np.exp(sigma) - 1) * self.lYs 
            - self.n * gammaln(np.exp(sigma))
            + self.a * sigma
            - self.b * np.exp(sigma)
            )

    def sample_alpha(self, curr_alpha):
        curr_log_alpha = np.log(curr_alpha)
        cand_log_alpha = normal(curr_log_alpha, scale = 5e-2)
        curr_lp = self.log_posterior_logalpha(curr_log_alpha)
        cand_lp = self.log_posterior_logalpha(cand_log_alpha)
        if np.log(uniform(1)) < cand_lp - curr_lp:
            return np.exp(cand_log_alpha)
        else:
            return np.exp(curr_log_alpha)
        pass
    
    def iter_sample(self):
        alpha = self.curr_alpha
        self.curr_iter += 1
        self.samples.alpha[self.curr_iter] = self.sample_alpha(alpha)
        return

    def __init__(self, Y, a, b):
        self.Y = Y
        self.lYs = np.log(Y).sum()
        self.n = Y.shape[0]
        self.a = a
        self.b = b
        return
    pass

class UnivariateGammaGammaMC(UnivariateResGammaGammaMC):
    @property
    def curr_beta(self):
        return self.samples.beta[self.curr_iter]
    def initialize_sampler(self, ns):
        self.samples = Samples(ns)
        self.samples.alpha[0] = 1.
        self.samples.beta[0] = 1.
        self.curr_iter = 0
        return
    def log_posterior_logalpha(self, sigma):
        return (
            + (np.exp(sigma) - 1) * self.lYs
            - self.n * gammaln(np.exp(sigma))
            + self.a * sigma
            - self.b * sigma
            + gammaln(self.n * np.exp(sigma) + self.c)
            - (self.n * np.exp(sigma) + self.c) * (self.Ys + self.d)
            )    
    def sample_beta(self, alpha):
        return gamma(
            shape = self.n * alpha + self.c, 
            scale = 1 / (self.Ys + self.d)
            )
    def iter_sample(self):
        alpha = self.curr_alpha
        self.curr_iter += 1
        self.samples.alpha[self.curr_iter] = self.sample_alpha(alpha)
        self.samples.beta[self.curr_iter]  = self.sample_beta(self.curr_alpha)
        return
    def __init__(self, Y, a, b, c, d):
        super().__init__(Y, a, b)
        self.Ys = Y.sum()
        self.c  = c
        self.d  = d
        return

class Adam(object):
    """ Adam optimizer """
    eps    : float = 1e-8
    rate   : float
    decay1 : float
    decay2 : float
    iter   : int

    theta    : np.ndarray # current value for theta
    momentum : np.ndarray # momentum
    sumofsqs : np.ndarray # sum of squares of past gradients

    def update(self):
        "performs adam optimization step on theta in-place"
        self.iter += 1
        dloss = self.dloss(self.theta)
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

    def initialization(
            self, 
            rate : float, 
            decay1 : float, 
            decay2 : float
            ):
        self.decay1 = decay1
        self.decay2 = decay2
        self.rate = rate
        self.iter = 0
        self.momentum = np.zeros(self.theta.shape[0])
        self.sumofsqs = np.zeros(self.theta.shape[0])
        return

    def optimize(self, niter):
        for _ in range(niter):
            self.update()
        return

    def __init__(self, theta, dloss, rate = 1e-3, decay1 = 0.9, decay2 = 0.999):
        self.dloss = dloss  # derivative of loss fn with respect to theta
        self.theta = theta  # initial value for theta
        self.initialization(rate, decay1, decay2)
        return

class UnivariateResGammaGammaVB(object):
    Y   : np.ndarray
    n   : int
    lYs : np.ndarray
    a   : float
    b   : float
    ns  : int

    def gradient(self, mu, tau):
        epsilon = normal(size = self.ns)
        ete = np.exp(tau) * epsilon
        alpha = np.exp(mu + ete)
        grad_alpha = (
            self.lYs
            - self.n * digamma(alpha)
            + (self.a - 1) / alpha
            - self.b
            )
        dmu = (grad_alpha * alpha).mean() - 1
        dtau = (grad_alpha * alpha * ete).mean() - 1 - tau
        return np.array((dmu, dtau))

    def __init__(self, Y, a, b, ns = 20):
        self.Y = Y
        self.n = Y.shape[0]
        self.lYs = np.log(Y).sum()
        self.a = a
        self.b = b
        self.ns = ns
        return
    pass

class UnivariateGammaGammaVB(UnivariateResGammaGammaVB):
    def gradient(self, mu, tau):
        epsilon = normal(size = self.ns)
        ete = np.exp(tau) * epsilon
        alpha = np.exp(mu + ete)
        grad_alpha = (
            self.lYs
            - self.n * digamma(alpha)
            + (self.a - 1) / alpha
            - self.b
            + digamma(self.n * alpha + self.c) * self.n
            - self.n * np.log(self.Ys + self.d)
            )
        dmu = (grad_alpha * alpha).mean() - 1
        dtau = (grad_alpha * alpha * ete).mean() - 1 - tau
        return np.array((dmu, dtau))
    
    def __init__(self, Y, a, b, c, d, ns = 20):
        super().__init__(Y, a, b, ns)
        self.Ys = Y.sum()
        self.c = c
        self.d = d
        return
    pass

if __name__ == '__main__':
    # gammavars = (8., 4.)
    gammavars = (5., 1.)
    Y = gamma(shape = gammavars[0], scale = 1 / gammavars[1], size = 500)
    abcd = (1.,1.,1.,1.)
    
    # mc = UnivariateResGammaGammaMC(Y, a, b)
    # mc.sample(5000)
    # vb = UnivariateResGammaGammaVB(Y, a, b)

    # mc = UnivariateGammaGammaMC(Y, *abcd)
    # vb = UnivariateGammaGammaVB(Y, *abcd)
    vb = UnivariateResGammaGammaVB(Y, *abcd[:2])
    def dloss(theta):
        return -vb.gradient(*theta)
    
    adam = Adam(
        np.array((0.,0.)), 
        dloss, 
        rate = 1e-3, 
        decay1 = 0.9, 
        decay2 = 0.999,
        )
    print('Initial: E[Y]: {}, Var[Y]: {}'.format(
        gammavars[0] / gammavars[1], 
        gammavars[0] / gammavars[1]**2
        ))
    print('mu: {}, tau: {}, E[Y]: {}, Var[Y]: {}'.format(
        *adam.theta, 
        np.exp(adam.theta[0] + np.exp(2 * adam.theta[1])/2),
        np.exp(2 * adam.theta[0] + np.exp(2 * adam.theta[1]) * 
               (np.exp(np.exp(2 * adam.theta[1])) - 1))
        ))
    print('Optimizing 1000 intervals')
    for _ in range(20):
        adam.optimize(1000)
        print('mu: {}, tau: {}, E[Y]: {}, Var[Y]: {}'.format(
            *adam.theta, 
            np.exp(adam.theta[0] + np.exp(2 * adam.theta[1])/2),
            np.exp(2 * adam.theta[0] + np.exp(2 * adam.theta[1]) * 
                (np.exp(np.exp(2 * adam.theta[1])) - 1))
            ))
# EOF