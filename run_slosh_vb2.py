from __future__ import division
import sys, os, glob, re
import numpy as np
import pandas as pd
import multiprocessing as mp
import sqlite3 as sql
from io import BytesIO
from collections import namedtuple
from numpy.random import uniform
import pickle as pkl

from energy import limit_cpu
from data import Data_From_Raw
# import varbayes as vb
from model_spypprg_vb import Chain, Result
import posterior as post

raw_path  = './datasets/slosh/filtered_data.csv.gz'
out_sql   = './datasets/slosh/results.sql'
out_table = 'energy'

def run_slosh(
        path_in,   path_out, 
        delta_out, cluster_out, 
        quantile, 
        eta,       discount,
        ):
    slosh = pd.read_csv(path_in, compression = 'gzip')
    slosh_ids = slosh.T[:8].T
    slosh_obs = slosh.T[8:].values.astype(np.float64)
    data = Data_From_Raw(slosh_obs, decluster = False, quantile = quantile)
    model = Chain(
        data = data, 
        concentration = eta, 
        discount = discount, 
	max_clusters = 100,
        gibbs_samples = 1000,
        )
    model.sample(10000)
    out = BytesIO()
    model.write_to_disk(out)
    res = Result(out)
    deltas = model.samples.delta
    # alphas = model.generate_conditional_posterior_alphas()
    smat = post.similarity_matrix(deltas)
    graph = post.minimum_spanning_trees(smat)
    g     = pd.DataFrame(graph)
    g = g.rename(columns = {0 : 'node1', 1 : 'node2', 2 : 'weight'})
    d = {
        'ids'    : slosh_ids,
        'obs'    : slosh_obs,
        'deltas' : deltas,
        'smat'   : smat,
        'graph'  : g,
        }
    with open(path_out, 'wb') as file:
        pkl.dump(d, file)
    pd.DataFrame(
        deltas.T, 
        columns = ['X{:03}'.format(i) for i in range(data.nDat)],
        ).to_csv(delta_out, index = False)
    pd.DataFrame({
        'obs' : np.arange(model.N),
        'pre' : post.emergent_clusters_pre(model),
        'pos' : post.emergent_clusters_post(model),
        }).to_csv(cluster_out, index = False)
    return

def run_slosh_for_nclusters(data, dataname, concentrations, discounts, verbose = False):
    ClusterCount = namedtuple('ClusterCount', 'eta discount ncluster')
    counts = []
    for eta in concentrations:
        for discount in discounts:
            model = Chain(data, concentration = eta, discount = discount)
            model.sample(50000, verbose = verbose)
            out = BytesIO()
            model.write_to_disk(out, 40000, 10)
            res = Result(out)
            nclusters = pd.DataFrame(res.samples.delta).nunique(axis = 1).mean()
            counts.append(ClusterCount(eta, discount, nclusters))
    pd.DataFrame(counts).to_csv('./test/ncluster_mc_{}.csv'.format(dataname))
    return

def instantiate_data(path, quantile):
    slosh = pd.read_csv(path, compression = 'gzip')
    slosh_obs = slosh.T[8:].values.astype(np.float64)
    return Data_From_Raw(slosh_obs, decluster = False, quantile = quantile)

concs_small = [0.001, 0.01, 0.1, 0.2]
discs_small = [0.001, 0.01, 0.1, 0.2]

concs_large = [0.1, 0.5, 2.0]
discs_large = [0.1, 0.2, 0.3]

concs_xlarg = [1.0, 10.0]
discs_xlarg = [0.2, 0.35, 0.5]

run = {
    # 't90' : True,
    'ltd' : True,
    'xpt' : True,
    'apt' : True,
    'emg' : True,
    # 'loc' : True,
    'del' : True,
    'nyc' : True,
    }
path_in_base  = './datasets/slosh/slosh_{}_data.csv.gz'
path_out_base = './datasets/slosh/slosh_{}_vb2.pkl'
clus_out_base = './datasets/slosh/slosh_{}_vb2_clusters.csv'
delt_out_base = './datasets/slosh/slosh_{}_vb2_delta.csv'
args = {
    't90' : {
        'path_in'     : path_in_base.format('t90'),
        'path_out'    : path_out_base.format('t90'),
        'cluster_out' : clus_out_base.format('t90'),
        'delta_out'   : delt_out_base.format('t90'),
        'quantile'    : 0.90,
        'concs'       : concs_xlarg,
        'discs'       : discs_xlarg,
        },
    'ltd' : {
        'path_in'     : path_in_base.format('ltd'),
        'path_out'    : path_out_base.format('ltd'),
        'cluster_out' : clus_out_base.format('ltd'),
        'delta_out'   : delt_out_base.format('ltd'),
        'quantile'     : 0.95, 
        'concs'       : concs_small,
        'discs'       : discs_small,
        },
    'xpt' : {
        'path_in'     : path_in_base.format('xpt'),
        'path_out'    : path_out_base.format('xpt'),
        'cluster_out' : clus_out_base.format('xpt'),
        'delta_out'   : delt_out_base.format('xpt'),
        'quantile'     : 0.95, 
        'concs'       : concs_small,
        'discs'       : discs_small,
        },
    'apt' : {
        'path_in'     : path_in_base.format('apt'),
        'path_out'    : path_out_base.format('apt'),
        'cluster_out' : clus_out_base.format('apt'),
        'delta_out'   : delt_out_base.format('apt'),
        'quantile'     : 0.95, 
        'concs'       : concs_small,
        'discs'       : discs_small,
        },
    'emg' : {
        'path_in'     : path_in_base.format('emg'),
        'path_out'    : path_out_base.format('emg'),
        'cluster_out' : clus_out_base.format('emg'),
        'delta_out'   : delt_out_base.format('emg'),
        'quantile'     : 0.95, 
        'concs'       : concs_small,
        'discs'       : discs_small,
        },
    'loc' : {
        'path_in'     : path_in_base.format('loc'),
        'path_out'    : path_out_base.format('loc'),
        'cluster_out' : clus_out_base.format('loc'),
        'delta_out'   : delt_out_base.format('loc'),
        'quantile'     : 0.95, 
        'concs'       : concs_xlarg,
        'discs'       : discs_xlarg,
        },
    'del' : {
        'path_in'     : path_in_base.format('del'),
        'path_out'    : path_out_base.format('del'),
        'cluster_out' : clus_out_base.format('del'),
        'delta_out'   : delt_out_base.format('del'),
        'quantile'     : 0.90, 
        'concs'       : concs_small,
        'discs'       : discs_small,
        },
    'nyc' : {
        'path_in'     : path_in_base.format('nyc'),
        'path_out'    : path_out_base.format('nyc'),
        'cluster_out' : clus_out_base.format('nyc'),
        'delta_out'   : delt_out_base.format('nyc'),
        'quantile'     : 0.90, 
        'concs'       : concs_small,
        'discs'       : discs_small,
        },
    }

if __name__ == '__main__':
    limit_cpu()
    # for dataset in run:
    #     if run[dataset]:
    #         run_slosh(**args[dataset])
    for dataset in run.keys(): 
        if run[dataset]:
            data = instantiate_data(
                args[dataset]['path_in'], args[dataset]['quantile'],
                )
            run_slosh_for_nclusters(
                data, dataset, 
                args[dataset]['concs'], args[dataset]['discs'],
                True,
                )
    # run_slosh(**{**args['del'], 'eta' : 0.01, 'discount' : 0.01})

# EOF
