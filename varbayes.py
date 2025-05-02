import numpy as np
import numpy.typing as npt
import pandas as pd
import projgamma as pg
from scipy.special import digamma, polygamma, softmax, logit, expit
from scipy.stats import norm as normal
from samplers import bincount2D_vectorized
from data import Data, euclidean_to_psphere, euclidean_to_hypercube
import matplotlib.pyplot as plt
import silence_tensorflow.auto
import tensorflow as tf
import tensorflow_probability as tfp
import time
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
from cUtility import pityor_cluster_sampler
from collections import namedtuple, deque
from tfprojgamma import ProjectedGamma
from numpy.random import beta
from projgamma import pt_logd_projgamma_my_mt_inplace_unstable,                 \
    logd_projgamma_my_mt_inplace_unstable
from samplers import py_sample_chi_bgsb_fixed, py_sample_cluster_bgsb_fixed,    \
    pt_py_sample_cluster_bgsb_fixed

def stickbreak(nu : npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
        Stickbreaking cluster probability
        nu : (S x (J - 1))
    """
    lognu = np.log(nu)
    log1mnu = np.log(1 - nu)

    S = nu.shape[0]; J = nu.shape[1] + 1
    out = np.zeros((S,J))
    out[:,:-1] += lognu
    out[:, 1:] += np.cumsum(log1mnu, axis = -1)
    return np.exp(out)

def stickbreak_tf(nu):
    batch_ndims = len(nu.shape) - 1
    cumprod_one_minus_nu = tf.math.cumprod(1 - nu, axis = -1)
    one_v = tf.pad(nu, [[0, 0]] * batch_ndims + [[0, 1]], "CONSTANT", constant_values=1)
    c_one = tf.pad(cumprod_one_minus_nu, [[0, 0]] * batch_ndims + [[1, 0]], "CONSTANT", constant_values=1)
    return one_v * c_one

class SurrogateVars(object):
    def init_vars(self, J, D, conc, disc, dtype):
        self.nu_mu    = tf.Variable(
            normal.ppf(1 / np.arange(2, J + 1)[::-1]), dtype = dtype, name = 'nu_mu',
            )
        self.nu_sd    = tf.Variable(
            tf.ones([J-1], dtype = dtype) * -3., name = 'nu_sd',
            )
        self.alpha_mu = tf.Variable(
            tf.random.normal([J,D], dtype = dtype), name = 'alpha_mu',
            )
        self.alpha_sd = tf.Variable(
            tf.random.normal([J,D], mean = -2, dtype = dtype), name = 'alpha_sd',
            )
        self.xi_mu    = tf.Variable(
            tf.random.normal([D],   dtype = dtype), name = 'xi_mu',
            )
        self.xi_sd    = tf.Variable(
            tf.random.normal([D],   dtype = dtype), name = 'xi_sd',
            )
        self.tau_mu   = tf.Variable(
            tf.random.normal([D],   dtype = dtype), name = 'tau_mu',
            )
        self.tau_sd   = tf.Variable(
            tf.random.normal([D],   dtype = dtype), name = 'tau_sd',
            )
        return
        
    def __init__(self, J, D, conc, disc, dtype = np.float64):
        self.init_vars(J, D, conc, disc, dtype)
        return
    
    pass

class SurrogateModel(object):
    def init_vars(self):
        self.vars = SurrogateVars(
            self.J, self.D, self.conc, self.disc, self.dtype,
            )
        return
    
    def init_model(self):
        self.model = tfd.JointDistributionNamed(dict(
            xi = tfd.Independent(
                tfd.LogNormal(
                    self.vars.xi_mu, tf.nn.softplus(self.vars.xi_sd),
                    ), 
                reinterpreted_batch_ndims = 1,
                ),
            tau = tfd.Independent(
                tfd.LogNormal(
                    self.vars.tau_mu, tf.nn.softplus(self.vars.tau_sd),
                    ), 
                reinterpreted_batch_ndims = 1,
                ),
            nu = tfd.Independent(
                tfd.LogitNormal(
                    self.vars.nu_mu, tf.nn.softplus(self.vars.nu_sd),
                    ), 
                reinterpreted_batch_ndims = 1,
                ),
            alpha = tfd.Independent(
                tfd.LogNormal(
                    self.vars.alpha_mu, tf.nn.softplus(self.vars.alpha_sd)
                    ),
                reinterpreted_batch_ndims = 2,
                ),
            ))
        return

    def sample(self, n):
        return self.model.sample(n)

    def __init__(self, J, D, concentration, discount, dtype = np.float64):
        self.J = J
        self.D = D
        self.conc = concentration
        self.disc = discount
        self.dtype = dtype
        self.init_vars()
        self.init_model()
        return
    
    pass

class WeightsInitializer(object):
    def clean_alpha_delta(self, alpha, delta):
        keep, delta[:] = np.unique(delta, return_inverse = True)
        return delta, alpha[keep]

    def posterior_means_of_cluster_weights(self, nj):
        shape1 = np.zeros(nj.shape[0] - 1)
        shape2 = np.zeros(nj.shape[0] - 1)
        shape1 += 1 - self.discount + nj[:-1]
        shape2 += self.concentration + np.arange(1, nj.shape[0]) * self.discount
        shape2 += (np.cumsum(nj[::-1])[::-1] - nj)[1:]
        # mean = shape1 / (shape1 + shape2)
        # var  = (shape1 * shape2) / ((shape1 + shape2)**2 * (shape1 + shape2 + 1))
        mean = digamma(shape1) - digamma(shape2)
        var  = polygamma(1, shape1) + polygamma(1, shape2)
        sd   = np.sqrt(var)
        return mean, sd
    
    def invsoftplus(self, y):
        return np.log(np.exp(y) - 1.)

    def set_weights(self):
        alpha  = np.exp(self.surrogate.vars.alpha_mu.numpy())
        J, D   = alpha.shape
        delta  = np.random.choice(J, size = self.data.nDat)
        N      = delta.shape[0]
        beta   = np.ones((J,D))
        loglik = np.zeros((N, J))
        for _ in range(self.niter):
            loglik[:] = 0.
            logd_projgamma_my_mt_inplace_unstable(loglik, self.Yp, alpha, beta)
            unifs = np.random.uniform(size = delta.shape[0])
            delta = pityor_cluster_sampler(
                delta, loglik, unifs, self.concentration, self.discount,
                )
            delta, alpha = self.clean_alpha_delta(alpha, delta)
            alpha = np.vstack((
                alpha, 
                np.random.lognormal(
                    mean = np.log(alpha).mean(axis = 0), 
                    sigma = np.max((np.log(alpha).std(), 2)), 
                    size = (J - alpha.shape[0], D)),
                ))
        nj = np.bincount(delta, minlength = J)
        mean, sd = self.posterior_means_of_cluster_weights(nj)
        self.surrogate.vars.alpha_mu.assign(np.log(alpha))
        self.surrogate.vars.nu_mu.assign(mean)
        self.surrogate.vars.nu_sd.assign(sd)
        return
    
    def __init__(self, data, surrogate, concentration, discount, nburn = 1000):
        self.data  = data
        self.Yp = euclidean_to_psphere(self.data.V, 10.)
        self.concentration = concentration
        self.discount = discount
        self.surrogate = surrogate
        self.niter = nburn
        return

class VarPYPG(object):
    """ 
        Variational Approximation of Pitman-Yor Mixture of Projected Gammas
        Constructed using TensorFlow, AutoDiff
    """
    start_time, end_time, time_elapsed = None, None, None

    def init_model(self):
        self.model = tfd.JointDistributionNamed(dict(
            xi = tfd.Independent(
                tfd.Gamma(
                concentration = np.full(self.D, self.a, self.dtype),
                rate = np.full(self.D, self.b, self.dtype),
                ),
                reinterpreted_batch_ndims = 1,
                ),
            tau = tfd.Independent(
                tfd.Gamma(
                    concentration = np.full(self.D, self.c, self.dtype),
                    rate = np.full(self.D, self.d, self.dtype),
                    ),
                reinterpreted_batch_ndims = 1,
                ),
            nu = tfd.Independent(
                tfd.Beta(
                    np.ones(self.J - 1, self.dtype) - self.discount, 
                    self.eta + np.arange(1, self.J) * self.discount
                    ),
                reinterpreted_batch_ndims = 1,
                ),
            alpha = lambda xi, tau: tfd.Independent(
                tfd.Gamma(
                    concentration = np.ones(
                        (self.J, self.D), self.dtype,
                        ) * tf.expand_dims(xi, -2),
                    rate = np.ones(
                        (self.J, self.D), self.dtype,
                        ) * tf.expand_dims(tau, -2),
                    ),
                reinterpreted_batch_ndims = 2,
                ),        
            obs = lambda alpha, nu: tfd.Sample(
                tfd.MixtureSameFamily(
                    mixture_distribution = tfd.Categorical(probs = stickbreak_tf(nu)),
                    components_distribution = ProjectedGamma(
                        alpha, np.ones((self.J, self.D), self.dtype)
                        ),
                    ),
                sample_shape = (self.N),
                ),
            ))
        _ = self.model.sample()
        
        def log_prob_fn(xi, tau, nu, alpha):
            return self.model.log_prob(
                xi = xi, tau = tau, nu = nu, alpha = alpha, obs = self.Yp,
                )
        self.log_prob_fn = log_prob_fn
        return

    def init_surrogate(self):
        self.surrogate = SurrogateModel(self.J, self.D, self.eta, self.discount, self.dtype)
        weightinit = WeightsInitializer(
            self.data, self.surrogate, self.eta, self.discount,
            )
        weightinit.set_weights()
        return
    
    def fit_advi(self, min_steps = 5000, max_steps = 100000,
                 relative_tolerance = 1e-6, seed = 1):
        optimizer = tf.optimizers.Adam(learning_rate = self.advi_learning_rate)
        concrit = tfp.optimizer.convergence_criteria.LossNotDecreasing(
            rtol = relative_tolerance, min_num_steps = min_steps,
            )
        self.start_time = time.time()
        losses = tfp.vi.fit_surrogate_posterior(
            target_log_prob_fn = self.log_prob_fn,
            surrogate_posterior = self.surrogate.model,
            optimizer = optimizer,
            convergence_criterion = concrit,
            sample_size = self.S,
            seed = seed,
            num_steps = max_steps,
            )
        # if num_steps and convergence criteria are both defined, then
        # num_steps becomes max_steps.  min_steps is defined in convergence
        # criteria object.
        self.end_time = time.time()
        self.time_elapsed = self.end_time - self.start_time
        return(losses)

    def __init__(
            self, 
            data, 
            eta = 0.2, 
            discount = 0.2, 
            prior_xi = (0.5, 0.5), 
            prior_tau = (2., 2.), 
            max_clusters = 100, # 200,
            dtype = np.float64,
            p = 10,
            advi_sample_size = 1,
            advi_learning_rate = 0.001,
            ):
        self.advi_learning_rate = advi_learning_rate
        self.data = data
        self.Yp = euclidean_to_psphere(self.data.V, p)
        self.J = max_clusters
        self.N = self.data.nDat
        self.D = self.data.nCol
        self.S = advi_sample_size
        self.a, self.b = prior_xi
        self.c, self.d = prior_tau
        self.eta = eta
        self.discount = discount
        self.dtype = dtype
        self.init_model()
        self.init_surrogate()
        return
    
    def generate_posterior_predictive_gamma(self, n = 5000):
        samples = self.surrogate.sample(n)
        alpha = samples['alpha'].numpy()
        N, J, D = alpha.shape
        nu = samples['nu'].numpy()
        delta = py_sample_cluster_bgsb_fixed(nu, np.zeros((N, J)))
        shape = alpha[np.arange(N), delta]
        return np.random.gamma(shape = shape, scale = 1, size = (N, D))
    
    def generate_posterior_predictive_hypercube(self, n = 5000):
        gammas = self.generate_posterior_predictive_gamma(n)
        return euclidean_to_hypercube(gammas)
    
    def generate_conditional_posterior_deltas_alphas(self, n = 1000):
        """ delta | y_i """
        samples = self.surrogate.sample(n)
        alpha   = samples['alpha'].numpy()
        nu      = samples['nu'].numpy()
        ll      = np.zeros((self.Yp.shape[0], n, alpha.shape[1]))
        pt_logd_projgamma_my_mt_inplace_unstable(
            ll, self.Yp, alpha, np.ones(alpha.shape),
            )
        logpi = np.zeros((n, nu.shape[1] + 1))
        logpi[:,:-1] += np.log(nu)
        logpi[:,1:]  += np.cumsum(np.log(1 - nu), axis = -1)
        lp    = ll + logpi[None] # log-posterior delta (unnormalized)
        po    = softmax(lp, axis = -1)
        po    = np.cumsum(po, axis = -1)
        t     = np.random.uniform(size = (po.shape[:-1]))
        delta = (po < t[...,None]).sum(axis = -1)
        return delta, alpha

    def generate_conditional_posterior_deltas(self, n = 1000):
        delta, alpha = self.generate_conditional_posterior_deltas_alphas(n = n)
        return delta

    def generate_conditional_posterior_alphas(self, n = 1000):
        """ \alpha | y_i """
        # samples = self.surrogate.sample(n)
        # alpha   = samples['alpha'].numpy()
        # nu      = samples['nu'].numpy()
        # ll      = np.zeros((self.Yp.shape[0], n, alpha.shape[1]))
        # pt_logd_projgamma_my_mt_inplace_unstable(
        #     ll, self.Yp, alpha, np.ones(alpha.shape),
        #     )
        # logpi = np.zeros((n, nu.shape[1] + 1))
        # logpi[:,:-1] += np.log(nu)
        # logpi[:,1:]  += np.cumsum(np.log(1 - nu), axis = -1)
        # lp    = ll + logpi[None] # log-posterior delta (unnormalized)
        # po    = softmax(lp, axis = -1)
        # po    = np.cumsum(po, axis = -1)
        # t     = np.random.uniform(size = (po.shape[:-1]))
        # delta = (po < t[...,None]).sum(axis = -1)
        delta, alpha = self.generate_conditional_posterior_deltas_alphas(n = n)
        
        out_alphas = np.empty((self.Yp.shape[0], n, alpha.shape[-1]))
        for s in range(n):
            out_alphas[:,s] = alpha[s][delta[:,s]]
        return out_alphas
    
    def generate_conditional_posterior_predictive_gammas(self, n = 500):
        alphas = self.generate_conditional_posterior_alphas(n)
        return np.random.gamma(alphas)
    
    def generate_conditional_posterior_predictive_hypercube(self, n = 500):
        gammas = self.generate_conditional_posterior_predictive_gammas(n)
        return euclidean_to_hypercube(gammas)
    
    pass

class ReducedSurrogateVars(SurrogateVars):
    def init_vars(self, J, D, dtype):
        self.alpha_mu = tf.Variable(
            tf.random.normal([J,D], dtype = dtype), name = 'alpha_mu',
            )
        self.alpha_sd = tf.Variable(
            tf.random.normal([J,D], dtype = dtype), name = 'alpha_sd',
            )
        self.xi_mu    = tf.Variable(
            tf.random.normal([D],   dtype = dtype), name = 'xi_mu',
            )
        self.xi_sd    = tf.Variable(
            tf.random.normal([D],   dtype = dtype), name = 'xi_sd',
            )
        self.tau_mu   = tf.Variable(
            tf.random.normal([D],   dtype = dtype), name = 'tau_mu',
            )
        self.tau_sd   = tf.Variable(
            tf.random.normal([D],   dtype = dtype), name = 'tau_sd',
            )
        return
    pass

class ReducedSurrogateModel(SurrogateModel):
    def init_model(self):
        self.model = tfd.JointDistributionNamed(dict(
            xi = tfd.Independent(
                tfd.LogNormal(
                    self.vars.xi_mu, tf.nn.softplus(self.vars.xi_sd),
                    ), 
                reinterpreted_batch_ndims = 1,
                ),
            tau = tfd.Independent(
                tfd.LogNormal(
                    self.vars.tau_mu, tf.nn.softplus(self.vars.tau_sd),
                    ), 
                reinterpreted_batch_ndims = 1,
                ),
            alpha = tfd.Independent(
                tfd.LogNormal(
                    self.vars.alpha_mu, tf.nn.softplus(self.vars.alpha_sd),
                    ), 
                reinterpreted_batch_ndims = 2,
                ),
            ))
        return
    
    def init_vars(self):
        self.vars = ReducedSurrogateVars(self.J, self.D, self.dtype)
        return 
    pass

class Samples(object):
    ci = None

    @property
    def curr_nu(self):
        return self.nu[self.ci % self.nMax]

    def update_nu(self, nu):
        self.ci += 1
        self.nu[self.ci % self.nMax] = nu
        return

    def __init__(self, nClust, nSamp, nKeep):
        self.nMax = nKeep
        # self.nu = np.zeros((nKeep, nSamp, nClust - 1))
        self.nu = np.zeros((nKeep, nClust - 1))
        self.ci = 0
        # self.nu[self.ci] = beta(0.5, 0.5, size = ((nSamp, nClust - 1)))
        self.nu[self.ci] = beta(0.5, 5, size = ((nClust - 1)))
        return

class MVarPYPG(VarPYPG):
    """ 
        Variational Approximation of Pitman-Yor Mixture of Projected Gammas
        with exact sampling of cluster membership / cluster weights
    """
    rate_placeholder = None
    
    @property
    def curr_nu(self):
        return self.samples.curr_nu

    def sample_delta(self, alpha, nu):
        scratch = np.zeros((self.D, self.J))
        pt_logd_projgamma_my_mt_inplace_unstable(scratch, self.Yp, alpha, self.rate_placeholder)
        return py_sample_cluster_bgsb_fixed(nu, scratch)
    
    def update_nu(self, alpha):
        """ Gibbs step update of nu given past nu, current alpha. """
        delta = self.sample_delta(alpha[np.random.choice(self.S)], self.curr_nu)
        nu = py_sample_chi_bgsb_fixed(delta, self.discount, self.eta, self.J)
        # delta = self.sample_delta(alpha, self.curr_nu)
        # nu = py_sample_chi_bgsb_fixed(delta, self.discount, self.eta, self.J)
        self.samples.update_nu(nu)
        pass

    def init_model(self):
        self.samples = Samples(self.J, self.S, self.nKeep)

        self.model = tfd.JointDistributionNamed(dict(
            xi = tfd.Independent(
                tfd.Gamma(
                    concentration = np.full(self.D, self.a, self.dtype),
                    rate = np.full(self.D, self.b, self.dtype),
                    )
                ),
            tau = tfd.Independent(
                tfd.Gamma(
                    concentration = np.full(self.D, self.c, self.dtype),
                    rate = np.full(self.D, self.d, self.dtype),
                    ),
                reinterpreted_batch_ndims = 1,
                ),
            alpha = lambda xi, tau: tfd.Independent(
                tfd.Gamma(
                    concentration = np.ones(
                        (self.J, self.D), self.dtype,
                        ) * tf.expand_dims(xi, -2),
                    rate = np.ones(
                        (self.J, self.D), self.dtype,
                        ) * tf.expand_dims(tau, -2),
                    ),
                reinterpreted_batch_ndims = 2,
                ),
            obs = lambda alpha : tfd.Sample(
                tfd.MixtureSameFamily(
                    mixture_distribution = tfd.Categorical(
                        probs = stickbreak_tf(self.curr_nu)
                        ),
                    components_distribution = ProjectedGamma(
                        alpha, np.ones((self.J, self.D), self.dtype)
                        ),
                    ),
                sample_shape = (self.N),
                ),
            ))
        _ = self.model.sample()

        def log_prob_fn(xi, tau, alpha):
            self.update_nu(alpha.numpy())
            return(self.model.log_prob(xi = xi, tau = tau, 
                                       alpha = alpha, nu = self.curr_nu))
        
        self.log_prob_fn = log_prob_fn
        return

    def init_surrogate(self):
        self.surrogate = ReducedSurrogateModel(self.J, self.D, self.dtype)
        return 

    def __init__(
            self, 
            data, 
            eta = 0.1, 
            discount = 0.1, 
            prior_xi = (0.5, 0.5), 
            prior_tau = (2., 2.), 
            max_clusters = 200,
            dtype = np.float64,
            nkeep = 3000,
            p = 10,
            advi_sample_size = 1,
            advi_learning_rate = 0.001,
            ):
        self.nKeep = nkeep
        super().__init__(
            data, eta, discount, prior_xi, prior_tau, 
            max_clusters, dtype, p, advi_sample_size,
            advi_learning_rate,
            )
        self.rate_placeholder = np.ones(
            (self.S, self.J, self.D), dtype = dtype,
            )
        self.samples = Samples(self.J, self.S, self.nKeep)
        return
    pass

class CVarPYPG(VarPYPG):
    rate_placeholder : np.ndarray

    @property
    def curr_expected_nu(self):
        return expit(self.surrogate.vars.nu_mu.numpy())

    def sample_delta(self, alpha, nu):
        loglik = np.zeros((self.S, self.N, self.J))
        pt_logd_projgamma_my_mt_inplace_unstable(loglik, self.Yp, alpha, self.rate_placeholder)
        return pt_py_sample_cluster_bgsb_fixed(log_likelihood=loglik, )

    def update_nu(self, alpha):
        delta = self.sample_delta(alpha.numpy().mean(axis = 0), self.curr_expected_nu)


    def fit_advi(self, min_steps = 5000, max_steps = 100000,
                 relative_tolerance = 1e-6, seed = 1):
        optimizer = tf.optimizers.Adam(learning_rate = self.advi_learning_rate)
        concrit = tfp.optimizer.convergence_criteria.LossNotDecreasing(
            rtol = relative_tolerance, min_num_steps = min_steps,
            )
        self.start_time = time.time()
        losses = tfp.vi.fit_surrogate_posterior(
            target_log_prob_fn = self.log_prob_fn,
            surrogate_posterior = self.surrogate.model,
            optimizer = optimizer,
            convergence_criterion = concrit,
            sample_size = self.S,
            seed = seed,
            num_steps = max_steps,
            trainable_variables = [
                self.surrogate.vars.alpha_mu, # (J x D)
                self.surrogate.vars.alpha_sd, # (J x D)
                self.surrogate.vars.xi_mu,    # (D)
                self.surrogate.vars.xi_sd,    # (D)
                self.surrogate.vars.tau_mu,   # (D)
                self.surrogate.vars.tau_sd,   # (D)
                ]
            )
        # if num_steps and convergence criteria are both defined, then
        # num_steps becomes max_steps.  min_steps is defined in convergence
        # criteria object.
        self.end_time = time.time()
        self.time_elapsed = self.end_time - self.start_time
        return(losses)

class MVNCholPrecisionTriL(tfd.TransformedDistribution):
    def __init__(self, loc, chol_precision_tril, name = None):
        super().__init__(
            distribution = tfd.Independent(
                tfd.Normal(
                    tf.zeros_like(loc), 
                    scale = tf.ones_like(loc),
                    reinterpreted_batch_ndims = 1,
                    ),
                bijector = tfb.Chain([
                    tfb.Shift(shift = loc), 
                    tfb.Invert(tfb.ScaleMatvecTriL(
                        scale_tril = chol_precision_tril, adjoint = True,
                        ))
                    ])
                ),
                name = name,
            )

    raw = pd.read_csv('./datasets/ivt_nov_mar.csv')
    dat = Data(raw, real_vars = np.arange(raw.shape[1]), quantile = 0.95)
    mod = VarPYPG(dat)
    mod.fit_advi()

def compute_shape(x_obs, x_loc, x_int, theta, epsilon):
    x_obs_reshape = [*theta.shape[:-1], *x_obs.shape]
    x_loc_reshape = [*theta.shape[:-1], *x_loc.shape]
    x_int_reshape = [*theta.shape[:-1], *x_int.shape]



    batch_ndims = len(theta.shape) - 2
    out = tf.zeros((*theta.shape[:-1], x_obs.shape[0], x_loc.shape[0]))

class VarPYPGR(VarPYPG):
    def init_model(self):
        self.model = tfd.JointDistributionNamed(dict(
            mu    = tfd.MVNCholPrecisionTriL(),
            Sigma = tfd.InverseWishart(),
            theta = tfd.MVNCholPrecisionTriL(),

        ))
    pass

if __name__ == '__main__':
    if False:
        np.random.seed(1)
        tf.random.set_seed(1)

        slosh = pd.read_csv(
            './datasets/slosh/filtered_data.csv.gz', 
            compression = 'gzip',
            )
        slosh_ids = slosh.T[:8].T
        slosh_obs = slosh.T[8:].T
        
        Result = namedtuple('Result','type ncol ndat time')
        sloshes = []

        for category in slosh_ids.Category.unique():
            idx = (slosh_ids.Category == category)
            ids = slosh_ids[idx]
            obs = slosh_obs[idx].values.T.astype(np.float64)
            dat = Data(obs, real_vars = np.arange(obs.shape[1]), quantile = 0.95)
            mod = MVarPYPG(dat)
            mod.fit_advi()

            sloshes.append(Result(category, dat.nCol, dat.nDat, mod.time_elapsed))
            print(sloshes[-1])
        
        pd.DataFrame(sloshes).to_csv('./datasets/slosh/times.csv', index = False)
        pass

# EOF