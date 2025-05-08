# Compute Distance / Divergence between Distributions
import numpy as np
import numpy.typing as npt
from itertools import repeat
from typing import Any
from collections import defaultdict
from math import log, sqrt
from sklearn.metrics import pairwise_distances
from multiprocessing import Pool, cpu_count
from energy import limit_cpu

def check_shape(data1 : npt.NDArray[Any], data2 : npt.NDArray[Any]) -> None:
    assert len(data1.shape) == 2
    assert len(data2.shape) == 2
    assert data1.shape[1] == data2.shape[1]
    return

def make_density_dictionary_hypercube(
        data : npt.NDArray[np.float64], 
        resolution : int,
        ) -> defaultdict[float]:
    n = data.shape[0]
    maxcol = np.argmax(data, axis = 1)
    _data = np.array([np.delete(data[i],maxcol[i]) for i in range(n)])
    _dint = np.floor(_data * resolution).astype(int)
    _dint_ = np.hstack((maxcol.reshape(-1,1), _dint))
    d = defaultdict(float)
    for iter in range(n):
        d[tuple(_dint_[iter].tolist())] += 1/n
    return d

def make_density_dictionary(
        data : npt.NDArray[np.float64], 
        resolution : float,
        ) -> defaultdict[float]:
    dint = np.ceil(data * resolution).astype(int)
    n = data.shape[0]
    d = defaultdict(float)
    for iter in range(n):
        d[tuple(dint[iter].tolist())] += 1/n
    return d

def kullbeck_liebler_divergence(
        data1 : npt.NDArray[np.float64], 
        data2 : npt.NDArray[np.float64], 
        resolution : int = 4,
        ) -> float:
    check_shape(data1,data2)
    d1 = make_density_dictionary_hypercube(data1, resolution)
    d2 = make_density_dictionary_hypercube(data2, resolution)

    divergence = 0.

    for key in d1.keys():
        divergence += d1[key] * (log(d1[key]) - log(d2[key]))

    return divergence

def hellinger_distance(
        data1 : npt.NDArray[np.float64], 
        data2 : npt.NDArray[np.float64], 
        resolution : int = 4,
        ) -> float:
    check_shape(data1, data2)
    d1 = make_density_dictionary_hypercube(data1, resolution)
    d2 = make_density_dictionary_hypercube(data2, resolution)

    combined_keys = set(d1.keys()).union(set(d2.keys()))

    sqdist = 0.

    for key in combined_keys:
        temp = sqrt(d1[key]) - sqrt(d2[key])
        sqdist += temp * temp

    return sqrt(sqdist) / sqrt(2)

def total_variation_distance(
        data1 : npt.NDArray[np.float64], 
        data2 : npt.NDArray[np.float64], 
        resolution : int = 4
        ) -> float:
    check_shape(data1, data2)
    d1 = make_density_dictionary_hypercube(data1, resolution)
    d2 = make_density_dictionary_hypercube(data2, resolution)

    combined_keys = set(d1.keys()).union(set(d2.keys()))

    m = 0.

    for key in combined_keys:
        m = max(m, abs(d1[key] - d2[key]))

    return m

def hypercube_deviance_inner(
        args : tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
        ) -> npt.NDArray[np.float64]:
    """
    Computes Euclidean distance between 
    """
    # face. indexes which face an obsv. falls on.
    face1 = np.argmax(args[0])              # target
    face2 = np.argmax(args[1], axis = -1)   # samples
    # Rotate samples into target's plane
    prime = args[1].copy()
    prime[:, face1] = 2. - prime[:,face2]
    prime[:, face2] = 1.
    # compute euclidean distance to target.
    prime -= args[0]
    prime *= prime
    return prime.sum(axis = -1)**0.5

def hypercube_deviance(
        data1 : npt.NDArray[np.float64],
        data2 : npt.NDArray[np.float64],
        ) -> npt.NDArray[np.float64]:
    """ 
    1: Verifies applicability of distance metric.
    2: Computes the upper bound on geodesic distance between samples from 
        data1 and data2.
    - For each obs from data1, rotates data2 into the same plane as obs, 
        and computes Euclidean distance between obs and data2^prime.
    """
    check_shape(data1, data2) # verify shapes compatible
    assert np.allclose(data1.max(axis = -1), 1.) # verify on hypercube
    assert np.allclose(data2.max(axis = -1), 1.)
    args = zip(data1, repeat(data2))
    with Pool(processes = cpu_count(), initializer = limit_cpu) as pool:
        res = pool.map(hypercube_deviance_inner, args)
        pool.close()
        pool.join()
    return np.array(list(res))

def cdf_distance(
        data1 : npt.NDArray[np.float64], 
        data2 : npt.NDArray[np.float64], 
        resolution : int = 10,
        ) -> float:
    """
    Contiuous Rank Probability Score - discretized
    """

    check_shape(data1, data2)
    d1 = make_density_dictionary_hypercube(data1, resolution)
    d2 = make_density_dictionary_hypercube(data2, resolution)

    nCol = data1.shape[1]
    nBin = int(nCol * (resolution**(nCol - 1)))

    # bins below per column iteration
    cBin = (resolution**np.array(range(nCol))).astype(int)[::-1]

    # Area of a Bin
    dL   = 1. / nBin

    # Unique Extant Bins -- No observations outside of this set.
    keys = sorted(list(set(d1.keys()).union(set(d2.keys()))))


    wassenstein2 = F1 = F2 = 0.
    bins_below = 0
    prev_bins_below = 0
    for key in keys:
        bins_below = (np.array(key, dtype = int) * cBin).sum()
        dL_below = (bins_below - prev_bins_below) / (nBin / nCol)
        # crps += dL * (F1 - (1 - F2)) * (F1 - (1 - F2))
        crps = dL_below * (F1 - F2) * (F1 - F2)

        F1   += d1[key]
        F2   += d2[key]

        dL_current = 1. / (nBin / nCol)
        wassenstein2 += dL_current * (F1 - (1 - F2)) * (F1 - (1 - F2))

        prev_bins_below = bins_below + 1
    return sqrt(wassenstein2)

def energy_score(
        prediction : npt.NDArray[np.float64], 
        target : npt.NDArray[np.float64],
        ) -> float:
    """ Computes Energy Score (Multivariate CRPS) between target and
        posterior predictive density """
    nSamp, nDat, nCol = prediction.shape

    GF = np.empty(nDat)
    PR = np.empty(nDat)

    for i in range(nDat):
        GF[i] = pairwise_distances(prediction[:,i], target[i].reshape(1,-1)).mean()

    for i in range(nDat):
        PR[i] = pairwise_distances(prediction[:,i]).mean()

    return 0.5 * PR.mean() - GF.mean()

if __name__ == '__main__':
    pass

# EOF