import numpy as np
import psutil
import os
from multiprocessing import Pool, cpu_count, pool as mcpool
from itertools import repeat
from sklearn.metrics import pairwise_distances
from .hypercube_deviance import hcdev

def hypercube_distance_unsummed(args):
    return pairwise_distances(args[0], args[1].reshape(1,-1), metric = hcdev)
def hypercube_distance(args):
    return pairwise_distances(args[0], args[1].reshape(1,-1), metric = hcdev).sum()
def hypercube_distance_mean(args):
    return pairwise_distances(args[0], args[1], metric = hcdev).mean()
def euclidean_distance_unsummed(args):
    return pairwise_distances(args[0], args[1].reshape(1,-1))
def manhattan_distance_unsummed(args):
    return pairwise_distances(args[0], args[1].reshape(1,-1), metric = 'manhattan')

distance_metrics = {
    'euclidean' : euclidean_distance_unsummed,
    'hypercube' : hypercube_distance_unsummed,
    'manhattan' : manhattan_distance_unsummed,
    }

def euclidean_distance_mean(args):
    return pairwise_distances(args[0], args[1]).mean()

def euclidean_dmat(args):
    return pairwise_distances(args[0], args[1])

def hypercube_dmat(args):
    return pairwise_distances(args[0], args[1], metric = hcdev)

def prediction_pairwise_distance(prediction):
    n = prediction.shape[0]
    res = map(hypercube_distance, zip(repeat(prediction), prediction))
    return np.array(list(res)).sum() / (n * n)

def target_pairwise_distance(args):
    prediction, target = args
    n = prediction.shape[0]
    return hypercube_distance((prediction, target)) / n

def limit_cpu():
    p = psutil.Process(os.getpid())
    if os.name == 'nt':
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    elif os.name == 'posix':
        p.nice(10)
    else:
        pass
    return

def energy_score_inner(predictions, targets):
    pool = Pool(processes = cpu_count(), initializer = limit_cpu)
    res1 = pool.map(prediction_pairwise_distance, predictions)
    res2 = pool.map(target_pairwise_distance, zip(predictions, targets))
    pool.close()
    pool.join()
    del pool
    return np.array(list(res2)) - 0.5 * np.array(list(res1))

def energy_score(predictions, targets):
    return energy_score_inner(predictions, targets).mean()

def energy_score_full(predictions, targets):
    pool = Pool(processes = cpu_count(), initializer = limit_cpu)
    res1 = prediction_pairwise_distance(predictions) # same for all elements.  do once.
    res2 = pool.map(target_pairwise_distance, zip(repeat(predictions), targets))
    pool.close()
    pool.join()
    del pool
    return np.array(list(res2)).mean() - 0.5 * res1

def postpred_loss_full(predictions, targets):
    pvari = np.cov(predictions.T).trace()
    pdev2 = ((targets - predictions.mean(axis = 0))**2).sum(axis = 1).mean()
    return pvari + pdev2

def energy_score_full_sc(predictions, targets):
    res1 = prediction_pairwise_distance(predictions)
    res2 = map(target_pairwise_distance, zip(repeat(predictions), targets))
    return np.array(list(res2)).mean() - 0.5 * res1

def intrinsic_energy_score(dataset):
    pool = Pool(processes = cpu_count(), initializer = limit_cpu)
    res1 = prediction_pairwise_distance(dataset) # same for all elements of df.  only do once.
    res2 = pool.map(target_pairwise_distance, zip(repeat(dataset),dataset))
    pool.close()
    pool.join()
    del pool
    return np.array(list(res2)).mean() - 0.5 * res1

def postpred_loss_single(predicted, empirical):
    """
    predicted:  (nSamp x nDat x nCol)
    empirical:  (nDat x nCol)
    """
    pmean = predicted.mean(axis = 0)
    pdiff = pmean - empirical
    bias  = (pdiff * pdiff).sum(axis = 1)
    pdevi = predicted - pmean
    pvari = np.empty(empirical.shape[0])
    for d in range(empirical.shape[0]):
        pvari[d] = np.trace(np.cov(pdevi[:,d].T))
    return bias + pvari

def hypercube_distance_matrix(predictions, targets, pool):
    res = pool.map(hypercube_distance_unsummed, zip(repeat(predictions), targets))
    return np.array(list(res))

def euclidean_distance_matrix(predictions, targets, pool):
    res = pool.map(euclidean_distance_unsummed, zip(repeat(predictions), targets))
    return np.array(list(res))

def manhattan_distance_matrix(predictions, targets, pool):
    res = pool.map(manhattan_distance_unsummed, zip(repeat(predictions), targets))
    return np.array(list(res))

def mixed_distance(real_pred, real_targ, cat_pred, cat_targ, pool):
    """
    predictions, targets = named tuples (with elements V, W)
    x.V = hypersphere projection
    x.W = one-hot encoding of categorical vars (alternatively, probability of class membership)
    ------------
    returns: (n_t, n_p, 2) np.ndarray, with:
        (x,y,0) : V distance (hypercube)
        (x,y,1) : W distance (euclidean)
    """
    hyp = hypercube_distance_matrix(real_pred, real_targ, pool)
    euc = euclidean_distance_matrix(cat_pred, cat_targ, pool)
    return np.vstack((hyp, euc))

def mixed_energy_score(real_pred, real_targ, cat_pred, cat_targ, pool):
    phyp, peuc = mixed_distance(real_pred, real_pred, cat_pred, cat_pred, pool)
    thyp, teuc = mixed_distance(real_pred, real_targ, cat_pred, cat_targ, pool)
    
    return (phyp + peuc).mean() - 0.5 * (thyp + teuc).mean()

def euclidean_dmat_per_obs(conditional, postpred, pool):
    """
    conditional : (n, s_c, d)  in R_+^d
    postpred    : (s_p, d)     in R_+^d
    pool        : multiprocessing.Pool
    """
    cshape = conditional.shape; pshape = postpred.shape
    args = zip(repeat(postpred), conditional.reshape(-1, cshape[-1]))
    res = np.array(list(pool.map(euclidean_distance_unsummed, args)))
    return res.reshape(cshape[0], cshape[1], pshape[0])

def hypercube_dmat_per_obs(conditional, postpred, pool):
    """
    conditional : (n, s_c, d)  in S_{infty}^{d-1}
    postpred    : (s_p, d)     in S_{infty}^{d-1}
    pool        : multiprocessing.Pool
    """
    cshape = conditional.shape; pshape = postpred.shape
    args = zip(repeat(postpred), conditional.reshape(-1, cshape[-1]))
    res = np.array(list(pool.map(hypercube_distance_unsummed, args)))
    return res.reshape(cshape[0], cshape[1], pshape[0])

def manhattan_dmat_per_obs(conditional, postpred, pool):
    cshape = conditional.shape; pshape = postpred.shape
    args = zip(repeat(postpred), conditional.reshape(-1, cshape[-1]))
    res = np.array(list(pool.map(manhattan_distance_unsummed, args)))
    return res.reshape(cshape[0], cshape[1], pshape[0])

def logkernel_gaussian(distance, bandwidth):
    lk = (
        - 0.5 * (distance / bandwidth)**2
        - np.log(bandwidth) 
        - 0.5 * np.log(2 * np.pi)
        )
    return lk

def logkernel_laplace(distance, bandwidth):
    lk = (
        - np.abs(distance / bandwidth)
        - np.log(2 * bandwidth)
        )

def kernel_gaussian(distance, bandwidth):
    # return np.exp(- 0.5 * (distance / bandwidth)**2) / (np.sqrt(2 * np.pi) * bandwidth)
    return np.exp(logkernel_gaussian(distance, bandwidth))

def kernel_laplace(distance, bandwidth):
    # return np.exp(- np.abs(distance / bandwidth)) / (2 * bandwidth)
    return np.exp(logkernel_laplace(distance, bandwidth))

kernels = {
    'gaussian' : kernel_gaussian,
    'laplace'  : kernel_laplace,
    }
logkernels = {
    'gaussian' : logkernel_gaussian,
    'laplace'  : logkernel_laplace,
    }

def kde_per_obs_inner(args):
    conditional, postpred, bandwidth, metric = args
    # def kde_per_obs_inner(conditional, postpred, bandwidth, metric, kernel = 'gaussian'):
    args = zip(repeat(postpred), conditional)
    res = map(distance_metrics[metric], args)
    distance = np.array(list(res))
    return kernels['gaussian'](distance, bandwidth).mean()

def kde_per_obs(conditional, postpred, bandwidth, metric, pool):
    """
    KDE per observation
    
    Args:
        conditional : (N, S1, D)
        postpred    : (S2, D)
        bandwidth   : (1)
        metric      : ('euclidean','hypercube')
        pool        : multiprocessing.Pool
    """
    args = zip(conditional, repeat(postpred), repeat(bandwidth), repeat(metric))
    res  = pool.map(kde_per_obs_inner, args)
    return np.array(list(res))

def multi_kde_per_obs_inner(pargs):
    # cond1 (D1)
    # pp1   (S2, D1)
    # band1 (1)
    # met1  ('euclidean','hypercube','manhattan')
    # cond2 (S1, D2)
    # pp2   (S2, D2)
    # band2 (1)
    # met2  ('euclidean','hypercube','manhattan')
    cond1, pp1, band1, met1, cond2, pp2, band2, met2 = pargs
    args1 = zip(repeat(pp1), cond1[None]) 
    args2 = zip(repeat(pp2), cond2)
    dis1  = np.array(list(map(distance_metrics[met1], args1))) # (1,  S2)
    dis2  = np.array(list(map(distance_metrics[met2], args2))) # (S1, S2)
    ld = logkernels['gaussian'](dis1, band1) + logkernels['gaussian'](dis2, band2)
    d = np.exp(ld).mean(axis = 0).mean()
    return d

def multi_kde_per_obs(cond1, pp1, band1, met1, cond2, pp2, band2, met2, pool):
    args = zip(
        cond1, repeat(pp1), repeat(band1), repeat(met1),
        cond2, repeat(pp2), repeat(band2), repeat(met2),
        )
    res = pool.map(multi_kde_per_obs_inner, args)
    return np.array(list(res))

def distance_to_point(a, b, metric):
    """
    a : array (n, d)
    b : vector (d)
    metric : string or func
    """
    return pairwise_distances(a, b.reshape(1,-1), metric = metric)

def real_energy_score(V, Vnew):
    pool = Pool(processes = cpu_count(), initializer = limit_cpu)
    res1 = np.array(list(pool.starmap(
                distance_to_point, zip(repeat(Vnew), Vnew, repeat(hcdev))
                )))
    res2 = np.array(list(pool.starmap(
                distance_to_point, zip(repeat(Vnew), V, repeat(hcdev))
                )))
    pool.close()
    pool.join()
    return res2.mean() - 0.5 * res1.mean()

def simp_energy_score(W, Wnew):
    pool = Pool(processes = cpu_count(), initializer = limit_cpu)
    res1 = np.array(list(pool.starmap(
                distance_to_point, zip(repeat(Wnew), Wnew, repeat('manhattan'))
                )))
    res2 = np.array(list(pool.starmap(
                distance_to_point, zip(repeat(Wnew), W, repeat('manhattan'))
                )))
    pool.close()
    pool.join()
    return res2.mean() - 0.5 * res1.mean()

def mixed_energy_score(V, W, Vnew, Wnew):
    pool = Pool(processes = cpu_count(), initializer = limit_cpu)
    res1 = np.array(list(pool.starmap(
                distance_to_point, zip(repeat(Vnew), Vnew, repeat(hcdev))
                )))
    res2 = np.array(list(pool.starmap(
                distance_to_point, zip(repeat(Vnew), V, repeat(hcdev))
                )))

    res3 = np.array(list(pool.starmap(
                distance_to_point, zip(repeat(Wnew), Wnew, repeat('manhattan'))
                )))
    res4 = np.array(list(pool.starmap(
                distance_to_point, zip(repeat(Wnew), W, repeat('manhattan'))
                )))
    pool.close()
    pool.join()
    return (res2 + res4).mean() - 0.5 * (res1 + res3).mean()

if __name__ == '__main__':

    # X = np.random.uniform(size = )
    



    pass

# EOF
