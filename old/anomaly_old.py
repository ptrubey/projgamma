""" 
Module for implementing anomaly detection algorithms.

Implements classic anomaly detection algorithms, as well as custom anomaly detection algorithms for extreme data.
"""
from inspect import Attribute
from xml.dom.minidom import Attr
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import re, os, argparse, glob, gc
# builtins explicitly called
from multiprocessing import pool as mcpool, cpu_count, Pool # get_context
from scipy.integrate import trapezoid
from scipy.special import gamma as gamma_func
from scipy.stats import gmean
from numpy.random import gamma, choice
from itertools import repeat
from collections import defaultdict
from functools import cached_property
from time import sleep
# Competing Anomaly Detection Algorithms
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from data import euclidean_to_hypercube, Projection
# Custom Modules
from energy import limit_cpu, euclidean_dmat_per_obs, hypercube_dmat_per_obs,       \
    hypercube_distance_matrix, euclidean_distance_matrix, manhattan_dmat_per_obs,   \
    mixed_energy_score, real_energy_score, simp_energy_score, kde_per_obs
from models import Results
np.seterr(divide = 'ignore')

EPS = np.finfo(float).eps 

# from classify import Classifier

def metric_auc(scores, actual):
    auroc = roc_auc_score(actual, scores)
    precision, recall, thresholds = precision_recall_curve(actual, scores)
    auprc = auc(recall, precision)
    pass
    # c = Classifier(scores, actual)
    # return (c.auroc, c.auprc)
    return(auroc, auprc)

class Anomaly(Projection):
    """ 
    Anomaly:

    Implements a variety of classic and experimental anomaly detection metrics.

    Usage:
        - Create composite class with (Result[model], Anomaly).
        - Instantiate.
        - Anomaly metrics will prefer to use existing distance metrics before generating new ones.
    Note:
        - Can declare multiprocessing.Pool first, so that *_distance_matrix will use 
            an existing pool rather than making a new one every time.
    """
    postpred_per_samp = 1

    # Parallelism
    pool = None
    def pools_open(self):
        # self.pool = get_context('spawn').Pool(
        self.pool = Pool(
            processes = (3 * cpu_count()) // 4, 
            initializer = limit_cpu,
            )
        return
    def pools_closed(self):
        self.pool.close()
        self.pool.join()
        del self.pool
        return

    @property
    def zeta_sigma(self):
        zetas = np.array([
            zeta[delta] 
            for zeta, delta 
            in zip(self.samples.zeta, self.samples.delta)
            ])
        try:
            sigmas = np.array([
                sigma[delta]
                for sigma, delta
                in zip(self.samples.sigma, self.samples.delta) 
                ])
        except AttributeError:
            sigmas = np.ones(zetas.shape)
        return zetas, sigmas

    @property
    def r(self):
        zetas, sigmas = self.zeta_sigma
        self.set_projection()
        r_shape = zetas[:,:,:self.nCol].sum(axis = 2)
        r_rate  = (sigmas[:,:,:self.nCol] * self.data.Yp[None,:,:]).sum(axis = 2)
        return gamma(r_shape, scale = 1 / r_rate)
    @property
    def rho(self):
        zetas, sigmas = self.zeta_sigma
        try:
            rho_shapes = zetas[:,:,self.nCol:] + self.data.W[None,:,:]
            rho_rates  = sigmas[:,:,self.nCol:]
            return gamma(rho_shapes, rho_rates)
        except AttributeError:
            return np.zeros((self.nSamp, self.nDat, 0))
    
    def energy_score(self):
        if hasattr(self.data, 'V') and hasattr(self.data, 'W'):
            Vnew = euclidean_to_hypercube(
                self.generate_posterior_predictive_gammas(self.postpred_per_samp)[:,:self.nCol]
                )
            Wnew = self.generate_posterior_predictive_spheres(self.postpred_per_samp)
            return mixed_energy_score(self.data.V, self.data.W, Vnew, Wnew)
        elif hasattr(self.data, 'V'):
            Vnew = euclidean_to_hypercube(
                self.generate_posterior_predictive_gammas(self.postpred_per_samp)[:,:self.nCol]
                )
            return real_energy_score(self.data.V, Vnew)
        elif hasattr(self.data, 'W'):
            Wnew = self.generate_posterior_predictive_spheres(self.postpred_per_samp)
            return simp_energy_score(self.data.W, Wnew)
        else:
            raise            

    ## Latent Distance per Observation (Summary)
    @property
    def euclidean_distance(self):
        mr = self.r.mean(axis = 0)
        mrho = self.rho.mean(axis = 0) 
        Y = np.hstack((mr[:,None] * self.data.Yp, mrho))
        return euclidean_distance_matrix(
            self.generate_posterior_predictive_gammas(), Y, self.pool,
            )
    @property
    def hypercube_distance(self):
        mr = self.r.mean(axis = 0)
        mrho = self.rho.mean(axis = 0)
        Y = np.hstack((mr[:,None] * self.data.Yp, mrho))
        V = euclidean_to_hypercube(Y)
        return hypercube_distance_matrix(
            self.generate_posterior_predictive_gammas(), V, self.pool,
            )
    @property
    def hypercube_distance_real(self):
        Vnew = euclidean_to_hypercube(
            self.generate_posterior_predictive_gammas()[:,:self.nCol],
            )
        return hypercube_distance_matrix(Vnew, self.data.V, self.pool)
    
    # Latent Distance per Observation (Sample)
    @property
    def sphere_distance_latent(self, V = None, W = None):
        pi_new = self.generate_posterior_predictive_spheres(10) # (s,d)
        if (V is None) and (W is None):
            pi_con = np.swapaxes(
                self.generate_conditional_posterior_predictive_spheres(), 0, 1,
                ) # (n, s, d)
        elif (V is not None) and (W is not None):
            pi_con = np.swapaxes(
                self.generate_conditional_posterior_predictive_sphere_new(V, W), 0, 1,
                )
        else:
            raise
        s = np.random.choice(pi_con.shape[1], pi_con.shape[1] // 2, False)
        return euclidean_dmat_per_obs(pi_con[:,s], pi_new, self.pool)
    @property
    def euclidean_distance_latent(self):
        R = self.generate_conditional_posterior_predictive_radii() # (s,n)
        Y1 = R[:,:,None] * self.data.V[None,:,:] # (s,n,d1),
        Y2 = self.generate_conditional_posterior_predictive_gammas()[:,:,self.nCol:] # (s,n,d2)
        Y_con = np.swapaxes(np.concatenate((Y1,Y2), axis = 2), 0, 1) # (n,s,d) 
        Y_new = self.generate_posterior_predictive_gammas()          # (s,d)
        s = np.random.choice(Y_con.shape[1], Y_con.shape[1]//2, False)
        return euclidean_dmat_per_obs(Y_con[:,s], Y_new, self.pool)
    @property
    def hypercube_distance_latent(self):
        R = self.generate_conditional_posterior_predictive_radii() # (s,n)
        Y1 = R[:,:,None] * self.data.V[None,:,:] # (s,n,d1),
        Y2 = self.generate_conditional_posterior_predictive_gammas()[:,:,self.nCol:] # (s,n,d2)
        Y_con = np.swapaxes(np.concatenate((Y1,Y2), axis = 2), 0, 1) # (n, s, d)
        V_con = np.array(list(map(euclidean_to_hypercube, Y_con)))
        V_new = euclidean_to_hypercube(self.generate_posterior_predictive_gammas())
        s = np.random.choice(V_con.shape[1], V_new.shape[1]//2, False)
        return hypercube_dmat_per_obs(V_con[:,s], V_new, self.pool)
    
    @cached_property
    def postpred_latent_euclidean(self):
        Y = self.generate_posterior_predictive_gammas()
        return Y
    @cached_property
    def latent_euclidean_bandwidth(self):
        Y = self.postpred_latent_euclidean
        YY = euclidean_dmat_per_obs(Y[None], Y, self.pool)
        return np.sqrt((YY**2).sum() / (2 * Y.shape[0] * (Y.shape[0] - 1)))
    @cached_property
    def postpred_latent_hypercube(self):
        return euclidean_to_hypercube(self.postpred_latent_euclidean)
    @cached_property
    def latent_sphere_bandwidth(self):
        P = self.generate_posterior_predictive_spheres(1)
        PP = manhattan_dmat_per_obs(P[None], P, self.pool)
        return np.sqrt((PP**2).sum() / (2 * P.shape[0] * (P.shape[0] - 1)))

    @cached_property
    def latent_hypercube_bandwidth(self):
        V = self.postpred_latent_hypercube
        VV = hypercube_dmat_per_obs(V[None], V, self.pool)
        return np.sqrt((VV**2).sum() / (2 * V.shape[0] * (V.shape[0] - 1) ))
    @cached_property
    def latent_mixed_bandwidth(self):
        V = euclidean_to_hypercube(
            self.generate_posterior_predictive_gammas(1)[:,:self.nCol]
            )
        P = self.generate_posterior_predictive_spheres(1)
        
        VV = hypercube_dmat_per_obs(V[None], V, self.pool)
        PP = euclidean_dmat_per_obs(P[None], P, self.pool)
        
        hV = np.sqrt((VV**2).sum() / (2 * V.shape[0] * (V.shape[0] - 1)))
        hP = np.sqrt((PP**2).sum() / (2 * P.shape[0] * (P.shape[0] - 1)))
        
        return (hV, hP)

    ## Classic Anomaly Metrics:
    def isolation_forest(self, V = None, W = None, **kwargs):
        """ Implements IsolationForest Method. Scores are arranged so larger = more anomalous """
        if hasattr(self.data, 'V') and hasattr(self.data, 'W'):
            dat = np.hstack((self.data.V, self.data.W))
        elif hasattr(self.data.V):
            dat = self.data.V
        elif hasattr(self.data.W):
            dat = self.data.W
        else:
            raise
        forest = IsolationForest().fit(dat)
        if (V is None) and (W is None):
            raw = forest.score_samples(dat)
        elif (V is not None) and (W is not None):
            datnew = np.hstack((V,W))
            raw = forest.score_samples(datnew)
        else:
            raise ValueError
        return raw.max() - raw + 1
    def local_outlier_factor(self, V = None, W = None, k = 5, **kwargs):
        """ Implements Local Outlier Factor.  k specifies the number of neighbors to fit to. """
        if hasattr(self.data, 'V') and hasattr(self.data, 'W'):
            dat = np.hstack((self.data.V, self.data.W))
        elif hasattr(self.data.V):
            dat = self.data.V
        elif hasattr(self.data.W):
            dat = self.data.W
        else:
            raise
        lof = LocalOutlierFactor(n_neighbors = k).fit(dat)
        if (V is None) and (W is None):
            raw = lof.negative_outlier_factor_.copy()
        elif (V is not None) and (W is not None):
            datnew = np.hstack((V, W))
            raw = lof.score_samples(datnew)
        else:
            raise
        return raw.max() - raw + 1
    def one_class_svm(self, V = None, W = None, **kwargs):        
        if hasattr(self.data, 'V') and hasattr(self.data, 'W'):
            dat = np.hstack((self.data.V, self.data.W))
        elif hasattr(self.data.V):
            dat = self.data.V
        elif hasattr(self.data.W):
            dat = self.data.W
        else:
            raise
        svm = OneClassSVM(gamma = 'auto').fit(dat)
        if (V is None) and (W is None):
            raw = svm.score_samples(dat)
        elif (V is not None) and (W is not None):
            datnew = np.hstack((V, W))
            raw = svm.score_samples(datnew)
        else: 
            raise
        return raw.max() - raw + 1

    ## Extreme Anomaly Metrics:
    def average_euclidean_distance_to_postpred(self, **kwargs):
        # return self.euclidean_distance.mean(axis = 1)
        return self.euclidean_distance_latent.mean(axis = (1,2))
    def average_hypercube_distance_to_postpred(self, **kwargs):
        # return self.hypercube_distance.mean(axis = 1)
        return self.hypercube_distance_latent.mean(axis = (1,2))
    def average_sphere_distance_to_postpred(self, **kwargs):
        return self.sphere_distance_latent.mean(axis = 1)
    def knn_hypercube_distance_to_postpred(self, k = 5, **kwargs):
        knn = np.array(list(map(np.sort, self.hypercube_distance)))[:,k, 0]
        if hasattr(self.data, 'V') and hasattr(self.data, 'W'):
            n = self.data.V.shape[0]
            p = self.data.V.shape[1] + self.data.W.shape[1]
        elif hasattr(self.data.V):
            n = self.data.V.shape[0]
            p = self.data.V.shape[1]
        elif hasattr(self.data.W):
            n = self.data.W.shape[0]
            p = self.data.W.shape[1]
        else:
            raise
        inv_scores =  (k / n) / (np.pi**((p-1)/2)/gamma_func((p-1)/2 + 1) * knn**(p-1))
        return 1 / inv_scores
    def knn_euclidean_distance_to_postpred(self, k = 5, **kwargs):
        knn = np.array(list(map(np.sort, self.euclidean_distance)))[:, k, 0]
        if hasattr(self.data, 'V') and hasattr(self.data, 'W'):
            n = self.data.V.shape[0]
            p = self.data.V.shape[1] + self.data.W.shape[1]
        elif hasattr(self.data.V):
            n = self.data.V.shape[0]
            p = self.data.V.shape[1]
        elif hasattr(self.data.W):
            n = self.data.W.shape[0]
            p = self.data.W.shape[1]
        else:
            raise
        inv_scores =  (k / n) / (np.pi**((p-1)/2)/gamma_func((p-1)/2 + 1) * knn**(p-1))
        return 1 / inv_scores
    def populate_cones(self, epsilon):
        postpred = euclidean_to_hypercube(self.generate_posterior_predictive_gammas())
        C_damex = (postpred > epsilon)
        cones = defaultdict(lambda: 1e-10)
        for row in C_damex:
            cones[tuple(row)] += 1 / postpred.shape[0]
        return cones
    def cone_density(self, epsilon = 0.5, **kwargs):
        cone_prob = self.populate_cones(epsilon)
        scores = np.empty(self.data.nDat)
        try:
            Y = euclidean_to_hypercube(
                    np.hstack((
                        self.samples.r.mean(axis = 0)[:, None] * self.data.V, 
                        self.samples.rho.mean(axis = 0)
                        ))
                    )
        except AttributeError:
            Y = self.data.V
        for i in range(self.nDat):
            scores[i] = cone_prob[tuple(Y[i] > epsilon)]
        return scores
    def hypercube_kernel_density_estimate(self, kernel = 'gaussian', **kwargs):
        # temporary code:
        h = self.latent_hypercube_bandwidth
        # h = gmean(self.hypercube_distance.ravel())
        if kernel == 'gaussian':
            return np.sqrt(2 * np.pi) * h / (np.exp(-(self.hypercube_distance / h)**2).mean(axis = (1,2)) + EPS)
        elif kernel == 'laplace':
            return 2 * h / (np.exp(-np.abs(self.hypercube_distance / h)).mean(axis = (1,2)) + EPS)
        else:
            raise ValueError('requested kernel not available')
        pass
    def euclidean_kernel_density_estimate(self, kernel = 'gaussian', **kwargs):
        # h = np.sqrt((self.euclidean_distance**2).mean()) * self.data.nDat**(-1/5)
        h = self.latent_euclidean_bandwidth
        if kernel == 'gaussian':
            return np.sqrt(2 * np.pi) * h / (np.exp(-(self.euclidean_distance / h)**2).mean(axis = (1,2)) + EPS)
        elif kernel == 'laplace':
            return 2 * h / (np.exp(-np.abs(self.euclidean_distance / h)).mean(axis = (1,2)) + EPS)
        else:
            raise ValueError('requested kernel not available')
        pass
    def latent_simplex_kernel_density_estimate(self, kernel = 'gaussian', **kwargs):
        """ computes mean kde for  """
        h = self.latent_sphere_bandwidth
        pi_con = np.swapaxes(self.generate_conditional_posterior_predictive_spheres(), 0, 1)
        pi_new = self.generate_posterior_predictive_spheres(10)
        inv_scores =  kde_per_obs(pi_con, pi_new, h, 'manhattan', self.pool)
        return 1 / (inv_scores + EPS)
    def latent_euclidean_kernel_density_estimate(self, kernel = 'gaussian', **kwargs):
        h = self.latent_euclidean_bandwidth
        R = self.generate_conditional_posterior_predictive_radii()   # (s,n)
        Y1 = R[:,:,None] * self.data.V[None,:,:]                     # (s,n,d1),
        Y2 = self.generate_conditional_posterior_predictive_gammas()[:,:,self.nCol:] # (s,n,d2)
        Y_con = np.swapaxes(np.concatenate((Y1,Y2), axis = 2), 0, 1) # (n,s,d) 
        Y_new = self.generate_posterior_predictive_gammas(self.postpred_per_samp)          # (s,d)
        inv_scores = kde_per_obs(Y_con, Y_new, h, 'euclidean', self.pool)
        return 1 / (inv_scores + EPS)
    def latent_hypercube_kernel_density_estimate(self, kernel = 'gaussian', **kwargs):
        h = self.latent_hypercube_bandwidth
        R = self.generate_conditional_posterior_predictive_radii() # (s,n)
        Y1 = R[:,:,None] * self.data.V[None,:,:] # (s,n,d1),
        Y2 = self.generate_conditional_posterior_predictive_gammas()[:,:,self.nCol:] # (s,n,d2)
        Y_con = np.swapaxes(np.concatenate((Y1,Y2), axis = 2), 0, 1) # (n, s, d)
        V_con = np.array(list(map(euclidean_to_hypercube, Y_con)))
        V_new = euclidean_to_hypercube(self.generate_posterior_predictive_gammas(self.postpred_per_samp))
        inv_scores = kde_per_obs(V_con, V_new, h, 'hypercube', self.pool)
        return 1 / (inv_scores + EPS)
    def mixed_latent_kernel_density_estimate(self, kernel = 'gaussian', **kwargs):
        h_real, h_simp = self.latent_mixed_bandwidth
        if kernel == 'gaussian':
            s1 = np.exp(-(self.hypercube_distance_real / h_real)**2).mean(axis = (1,2))
            # s2 = np.exp(-(self.sphere_distance_latent / h_simp)**2).mean(axis = (1,2))
            s2 = self.latent_simplex_kernel_density_estimate()
            return 1 / (s1 * s2 + EPS)
        elif kernel == 'laplace':
            s1 = np.exp(-np.abs(self.hypercube_distance_real / h_real)).mean(axis = (1,2))
            s1 = np.exp(-np.abs(self.sphere_distance_latent / h_simp)).mean(axis = (1,2))
            return 1 / (s1 * s2 + EPS)
        else:
            raise ValueError('requested kernel not available')
        pass
    def combined_knn_hypercube_distance_to_postpred(self, **kwargs):
        return self.knn_hypercube_distance_to_postpred(**kwargs) * self.data.R
    def combined_knn_euclidean_distance_to_postpred(self, **kwargs):
        return self.knn_euclidean_distance_to_postpred(**kwargs) * self.data.R
    def combined_cone_density(self, **kwargs):
        return self.cone_density(**kwargs) * self.data.R
    def combined_hypercube_kernel_density_estimate(self, **kwargs):
        return self.hypercube_kernel_density_estimate(**kwargs) * self.data.R
    def combined_euclidean_kernel_density_estimate(self, **kwargs):
        return self.euclidean_kernel_density_estimate(**kwargs) * self.data.R
    def combined_latent_simplex_kernel_density_estimate(self, **kwargs):
        return self.latent_simplex_kernel_density_estimate(**kwargs) * self.data.R
    def combined_latent_euclidean_kernel_density_estimate(self, **kwargs):
        return self.latent_euclidean_kernel_density_estimate(**kwargs) * self.data.R
    def combined_latent_hypercube_kernel_density_estimate(self, **kwargs):
        return self.latent_hypercube_kernel_density_estimate(**kwargs) * self.data.R
    def combined_mixed_latent_kernel_density_estimate(self, **kwargs):
        return self.mixed_latent_kernel_density_estimate(**kwargs) * self.data.R

    # scoring metrics
    @property
    def scoring_metrics(self):
        metrics = {
            'iso'    : self.isolation_forest,
            'lof'    : self.local_outlier_factor,
            'svm'    : self.one_class_svm,
            # 'aedp'   : self.average_euclidean_distance_to_postpred,
            # 'ahdp'   : self.average_hypercube_distance_to_postpred,
            'kedp'   : self.knn_euclidean_distance_to_postpred,
            'khdp'   : self.knn_hypercube_distance_to_postpred,
            'cone'   : self.cone_density,
            'ekde'   : self.euclidean_kernel_density_estimate,
            'hkde'   : self.hypercube_kernel_density_estimate,
            'lhkde'  : self.latent_hypercube_kernel_density_estimate,
            'lekde'  : self.latent_euclidean_kernel_density_estimate,
            'lskde'  : self.latent_simplex_kernel_density_estimate,
            'mlkde'  : self.mixed_latent_kernel_density_estimate,
            }
        return metrics
    def get_scores(self):
        metrics = self.scoring_metrics.keys()
        density_metrics = ['khdp','kedp','cone','hkde','ekde','lskde','lekde','lhkde','mlkde']
        out = pd.DataFrame()
        for metric in metrics:
            print('s' + '\b'*11 + metric.ljust(10), end = '')
            sleep(1)
            out[metric] = self.scoring_metrics[metric]().ravel()
            if hasattr(self.data, 'R'):
                if metric in density_metrics:
                    out['c' + metric] = out[metric] * self.data.R
        print('s' + '\b'*11 + 'Done'.ljust(10))
        return out
    def get_scoring_metrics(self):
        scores = self.get_scores()
        aucs = np.array([metric_auc(score, self.data.Y) for score in scores.values.T]).T
        metrics = pd.DataFrame(aucs, columns = scores.columns.values.tolist())
        metrics['Metric'] = ('AuROC','AuPRC')
        metrics['EnergyScore'] = self.energy_score()
        return metrics
    def get_scores_new(self):
        pass

def ResultFactory(model, path):
    class Result(Results[model], Anomaly):
        pass

    return Result(path)

def MixedResultFactory(path):
    if 'mdppprgln' in path:
        class Result(Results['mdppprgln'], Anomaly):
            pass
        return Result(path)
    elif 'mdppprg' in path:
        class Result(Results['mdppprg'], Anomaly):
            pass
        return Result(path)
    else: 
        raise ValueError('Wrong!')
    pass

def plot_log_inverse_scores(scores):
    plt.plot(np.sort(np.log(1/scores)))
    plt.show()
    return

def plot_log_inverse_scores_knn(scores):
    ord = np.argsort(scores.mean(axis = 1))
    plt.plot(np.log(1/scores[ord[::-1]]))
    plt.show()
    return

def argparser():
    p = argparse.ArgumentParser()
    p.add_argument('in_path')
    p.add_argument('out_path')
    return p.parse_args()

if __name__ == '__main__':
    results  = []
    basepath = './ad'
    datasets = ['cardio','cover','mammography','pima','satellite']
    resbases = {'mdppprgln' : 'results_*.pkl'}
    for model in resbases.keys():
        for dataset in datasets:
            files = glob.glob(os.path.join(basepath, dataset, resbases[model]))
            for file in files:
                results.append((model, file))
    metrics = []
    for result in results:
        print('Processing Result {}'.format(result[1]).ljust(80), end = '')
        extant_result = ResultFactory(*result)
        extant_result.p = 10.
        extant_result.pools_open()
        extant_metric = extant_result.get_scoring_metrics()
        extant_result.pools_closed()
        del extant_result
        extant_metric['path'] = result[1]
        metrics.append(extant_metric)
        gc.collect()
    
    df = pd.concat(metrics)
    df.to_csv('./ad/performance.csv')

    # path = './simulated/lnad/results_mdppprgln.pkl'
    # print('Processing Result {}'.format(path).ljust(80), end = '')
    # extant_result = ResultFactory('mdppprgln', path)
    # extant_result.p = 10
    # extant_result.pools_open()
    # scores = extant_result.get_scoring_metrics()
    # extant_result.pools_closed()
    # raise

# EOF   
