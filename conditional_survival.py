import numpy as np
import pandas as pd
import glob, os
from collections import defaultdict
from itertools import repeat
from energy import limit_cpu
from data import Data_From_Raw, scale_pareto, descale_pareto
from argparser import argparser_cs as argparser
import models, models_mpi

def condsurv_at_w(args):
    """args = (new_dims, new_vec, given_dims, given_vec, postpred)"""
    new_dims, new_vec, given_dims, given_vec, postpred = args
    w = np.empty(new_vec.shape[0] + given_vec.shape[0])
    w[given_dims] = given_vec
    w[new_dims]   = new_vec
    return (postpred / w).min(axis = 1).mean()

def condsurv_at_w_precomp(args):
    """ args = (w_new, postpred_new, min_thus_far) """
    min_new = (args[1] / args[0]).min(axis = 1)
    return np.vstack((min_new, args[2])).min(axis = 0).mean()

class Conditional_Survival(object):
    """ Computes 1,2 dimensional conditional survival curves given all other dimensions """
    def set_prediction_space(self, lower_scalar = 0.05, upper_scalar = 0.2):
        min = self.data.raw.min(axis = 0)
        max = self.data.raw.max(axis = 0)
        dif = max - min
        lower_bound = min - lower_scalar * dif # lower bound for prediction space
        upper_bound = max + upper_scalar * dif # upper bound for prediction space
        return np.vstack((lower_bound, upper_bound))

    def condsurv_2d(self, given_dims, given_vec, n_per_sample = 10):
        """ Conditional Survival of 2 dimensions given all others -- in original units """
        postpred = self.generate_posterior_predictive_hypercube(n_per_sample)
        new_dims = np.setdiff1d(np.arange(self.nCol), given_dims)
        prediction_space = self.set_prediction_space().T[new_dims]
        prediction_linspace = [np.linspace(*x, 100) for x in prediction_space]
        unscaled = np.array([(a,b) for a in prediction_linspace[0] for b in prediction_linspace[1]])
        new_dim_values = scale_pareto(unscaled, self.data.P[new_dims])
        args = zip(repeat(new_dims), new_dim_values, repeat(given_dims),
                                        repeat(given_vec), repeat(postpred))
        pool = mp.Pool(processes = os.cpu_count(), initializer = limit_cpu)
        res = pool.map(condsurv_at_w, args)
        pool.close()
        return np.hstack((unscaled, np.array(list(res)).reshape(-1,1)))

    def condsurv_1d(self, given_dims, given_vec, n_per_sample = 10):
        """ Conditional Survival of one dimension given all others """
        postpred = self.generate_posterior_predictive_hypercube(n_per_sample)
        new_dim = np.setdiff1d(np.arange(self.nCol), given_dims)
        prediction_space = self.set_prediction_space().T[new_dim]
        unscaled = np.linspace(*prediction_space, 100)
        new_dim_values = scale_pareto(unscaled, self.data.P[new_dim])
        args = zip(repeat(new_dim), new_dim_values, repeat(given_dims),
                                        repeat(given_vec), repeat(postpred))
        pool = mp.Pool(processes = os.cpu_count(), initializer = limit_cpu)
        res = pool.map(condsurv_at_w, args)
        pool.close()
        return np.hstack((unscaled.reshape(-1,1), np.array(list(res)).reshape(-1,1)))

    def condsurv_1d_at_quantile(
            self,
            given_dims,
            given_vec_quantile,
            prediction_bounds = (0.8, 1.),
            n_per_sample = 10,
            ):
        postpred = self.generate_posterior_predictive_hypercube(n_per_sample)
        new_dim  = np.setdiff1d(np.arange(self.nCol), given_dims)
        w_new    = postpred.T[new_dim].mean() / (1 - np.linspace(*prediction_bounds, 100, False))
        w_given  = postpred.T[given_dims].mean(axis = 1) / (1 - given_vec_quantile)
        min_thus_far = (postpred.T[given_dims].T / w_given).min(axis = 1)
        args = zip(w_new.reshape(-1,1), repeat(postpred.T[new_dim].reshape(-1,1)), repeat(min_thus_far))
        out  = np.array(list(map(condsurv_at_w_precomp, args)))
        return np.hstack((w_new.reshape(-1,1), out.reshape(-1,1) / min_thus_far.mean()))

    def condsurv_2d_at_quantile(
            self,
            given_dims,
            given_vec_quantile,
            prediction_bounds = (0.8, 1.),
            n_per_sample = 10,
            ):
        postpred = self.generate_posterior_predictive_hypercube(n_per_sample)
        new_dims = np.setdiff1d(np.arange(self.nCol), given_dims)
        w_given = postpred.T[given_dims].mean(axis = 1) / (1 - given_vec_quantile)
        f = lambda x: x / (1 - np.linspace(*prediction_bounds, 100, False))
        w_new = np.array(list(map(f, postpred.T[new_dims].mean(axis = 1)))).reshape(-1,2)
        w_new_grid = np.array([(a,b) for a in w_new.T[0] for b in w_new.T[1]])
        min_thus_far = (postpred.T[given_dims].T / w_given).min(axis = 1)
        args = zip(w_new_grid, repeat(postpred.T[new_dims].T), repeat(min_thus_far))
        out = np.array(list(map(condsurv_at_w_precomp, args)))
        return np.hstack((w_new_grid, out.reshape(-1,1) / min_thus_far.mean()))

    def load_raw(self, path):
        raw = pd.read_csv(path)
        self.data = Data_From_Raw(raw, True)
        return

    pass

Results = {**models.Results, **models_mpi.Results}

def ResultFactory(model, raw_path):
    class Result(Results[model[0]], Conditional_Survival):
        pass

    result = Result(model[1])
    result.load_raw(raw_path)
    return result

if __name__ == '__main__':
    args = argparser()
    model_types = sorted(Results.keys())
    models = []
    for model_type in model_types:
        mm = glob.glob(os.path.join(args.path, model_type, 'results*.db'))
        for m in mm:
            models.append((model_type, m))

    for model in models:
        pass

    r = ResultFactory(models[0], args.raw_path)

# EOF