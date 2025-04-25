# Compute Distance / Divergence between Distributions
import numpy as np
import pandas as pd
from collections import defaultdict, namedtuple
from math import log, sqrt
from data import to_hypercube
from sklearn.metrics import pairwise_distances

def check_shape(data1,data2):
    try:
        data1.shape[1] == data2.shape[1]
    except AssertionError:
        print('Data do not have same number of columns!')
        raise
    return

def make_density_dictionary_hypercube(data, resolution):
    n = data.shape[0]
    maxcol = np.argmax(data, axis = 1)
    _data = np.array([np.delete(data[i],maxcol[i]) for i in range(n)])
    _dint = np.floor(_data * resolution).astype(int)
    _dint_ = np.hstack((maxcol.reshape(-1,1), _dint))
    d = defaultdict(float)
    for iter in range(n):
        d[tuple(_dint_[iter].tolist())] += 1/n
    return d

def make_density_dictionary(data, resolution):
    dint = np.ceil(data * resolution).astype(int)
    n = data.shape[0]
    d = defaultdict(float)
    for iter in range(n):
        d[tuple(dint[iter].tolist())] += 1/n
    return d

def kullbeck_liebler_divergence(data1, data2, resolution = 4):
    check_shape(data1,data2)
    d1 = make_density_dictionary_hypercube(data1, resolution)
    d2 = make_density_dictionary_hypercube(data2, resolution)

    divergence = 0.

    for key in d1.keys():
        divergence += d1[key] * (log(d1[key]) - log(d2[key]))

    return divergence

def hellinger_distance(data1, data2, resolution = 4):
    check_shape(data1, data2)
    d1 = make_density_dictionary_hypercube(data1, resolution)
    d2 = make_density_dictionary_hypercube(data2, resolution)

    combined_keys = set(d1.keys()).union(set(d2.keys()))

    sqdist = 0.

    for key in combined_keys:
        temp = sqrt(d1[key]) - sqrt(d2[key])
        sqdist += temp * temp

    return sqrt(sqdist) / sqrt(2)

def total_variation_distance(data1, data2, resolution = 4):
    check_shape(data1, data2)
    d1 = make_density_dictionary_hypercube(data1, resolution)
    d2 = make_density_dictionary_hypercube(data2, resolution)

    combined_keys = set(d1.keys()).union(set(d2.keys()))

    m = 0.

    for key in combined_keys:
        m = max(m, abs(d1[key] - d2[key]))

    return m

def cdf_distance(data1, data2, resolution = 10):
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

def energy_score(prediction, target):
    """ Computes Energy Score (Multivariate CRPS) between target and
        posterior predictive density """
    nSamp, nDat, nCol = prediction.shape

    GF = np.empty(nDat)
    PR = np.empty(nDat)

    for i in range(nDat):
        #diff = prediction[:,i] - target[i]
        #GF[i] = np.sqrt((diff * diff).sum(axis = 1)).mean()
        GF[i] = pairwise_distances(prediction[:,i], target[i].reshape(1,-1)).mean()

    for i in range(nDat):
        PR[i] = pairwise_distances(prediction[:,i]).mean()

    return 0.5 * PR.mean() - GF.mean()

if __name__ == '__main__':
    import glob, os
    base_path = './output'
    base_models = ['fmix','mpg','dpmpg','dppgln','dp']

    Posterior = namedtuple('Posterior','path type name crps hell tv')
    posteriors = []

    for base_model in base_models:
        realized_models = glob.glob(os.path.join(base_path, base_model, 'postpred_*.csv'))
        emp = to_hypercube(
            pd.read_csv(os.path.join(base_path, base_model, 'empirical.csv')).values
            )

        for realized_model in realized_models:
            pp = to_hypercube(pd.read_csv(realized_model).values)
            posteriors.append(
                Posterior(
                    realized_model,
                    base_model,
                    os.path.splitext(os.path.split(realized_model)[1])[0],
                    # np.nan, # kullbeck_liebler_divergence(emp, pp),
                    energy_score(emp, pp),
                    hellinger_distance(emp, pp),
                    total_variation_distance(emp, pp),
                    )
                )

    print_string = '{}:{} : crps {}, Hellinger {}, TV {}'
    for post in posteriors:
        print(print_string.format(post.type, post.name, post.crps, post.hell, post.tv))
    df = pd.DataFrame([posterior[1:] for posterior in posteriors],
                        columns = ('type','name','crps','hellinger','tv'))
    df.to_csv('./output/distance.csv')

# EOF
#cython: boundscheck=False, wraparound=False, nonecheck=False
