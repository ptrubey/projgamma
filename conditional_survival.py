import numpy as np
import pandas as pd
import multiprocessing as mp
import os
from itertools import repeat, product

from energy import limit_cpu
from data import Data

def condsurv_helper(args):
    """ Helper function for conditional survival """
    tgz, ppz, mtf = args # target_z, postpred_z, mean_thus_far
    v_over_z = ppz[None] / tgz[:,None]  
    v_over_z[np.where(v_over_z > 1)] = 1
    vz_min = v_over_z.min(axis = -1)
    tot_min = np.minimum(vz_min, mtf)
    return np.hstack((tgz, tot_min.mean(axis = -1).reshape(-1,1)))

class Conditional_Survival(object):
    """ Computes conditional survival curves / surfaces given all other dimensions. """
    data : Data
    
    def condsurv_at_quantile_std(
            self,
            target_dims : np.ndarray,
            given_dims : np.ndarray,
            given_vec_quantile : np.ndarray,
            prediction_range = (0.001, 8, 500),
            n_per_sample = 10,
            obs = None,
            splits = 200,
            verbose = False,
            ):
        """ Computes conditional survival curves/surfaces given all other dimensions. """
        assert given_dims.shape[0] == given_vec_quantile.shape[0]
        if verbose:
            print('Target {}, Given {}, Obsv {}'.format(target_dims, given_dims, obs))
        postpred = self.generate_posterior_predictive_hypercube(
            n_per_sample = n_per_sample, obs = obs,
            )
        try:
            self.S
        except AttributeError:
            self.S = self.data.nCol
        ignored_dims = np.setdiff1d(
            np.arange(self.S), 
            np.union1d(target_dims, given_dims),
            )
        given_z = np.array([
            np.quantile(self.data.Z.T[dim], q = given_vec_quantile[i])
            for i, dim in enumerate(given_dims)
            ])
        pred_bounds = np.linspace(*prediction_range)
        targets = [pred_bounds for _ in range(target_dims.shape[0])]
        target_z = np.array(list(product(*targets)))
        given_z  = np.array([
            np.quantile(self.data.Z.T[given_dims[dim]], given_vec_quantile[dim])
            for dim in np.arange(given_dims.shape[0])
            ])
        conditioning_z = np.zeros((self.S))
        conditioning_z[given_dims] = given_z
        conditioning_z[ignored_dims] += 1e-10
        thus_far = postpred.T[np.union1d(given_dims, ignored_dims)].T           \
                    / conditioning_z[np.union1d(given_dims, ignored_dims)]
        thus_far[np.where(thus_far > 1)] = 1
        min_thus_far = (thus_far).min(axis = 1)

        nsplit = target_z.shape[0] // splits # arrays of size 200 per

        target_z_list = np.array_split(target_z, nsplit)
        args = zip(
            target_z_list, 
            repeat(postpred.T[target_dims].T), 
            repeat(min_thus_far),
            )
        
        with mp.Pool(processes = mp.cpu_count(), initializer = limit_cpu) as pool:
            out = np.vstack(list(pool.map(condsurv_helper, args)))
        
        out.T[-1] /= min_thus_far.mean()
        return out

if __name__ == '__main__':
    pass

# EOF
