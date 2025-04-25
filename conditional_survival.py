import numpy as np
import pandas as pd
import multiprocessing as mp
import os
from itertools import repeat, product

from energy import limit_cpu
from data import Data_From_Raw, scale_pareto, descale_pareto
from models import Results 

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

def condsurv_helper(args):
    """  """
    tgz, ppz, mtf = args # target_z, postpred_z, mean_thus_far
    v_over_z = ppz[None] / tgz[:,None]  
    v_over_z[np.where(v_over_z > 1)] = 1
    vz_min = v_over_z.min(axis = -1)
    tot_min = np.minimum(vz_min, mtf)
    return np.hstack((tgz, tot_min.mean(axis = -1).reshape(-1,1)))

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
        """ Conditional Survival of 2 dimensions given all others """
        postpred = self.generate_posterior_predictive_hypercube(n_per_sample)
        new_dims = np.setdiff1d(np.arange(self.nCol), given_dims)
        prediction_space = self.set_prediction_space().T[new_dims]
        prediction_linspace = [np.linspace(*x, 100) for x in prediction_space]
        unscaled = np.array([
            (a,b) 
            for a in prediction_linspace[0] 
            for b in prediction_linspace[1]
            ])
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

    def condsurv_1d_at_quantile_std(
            self,
            given_dims,
            given_vec_quantile,
            n_per_sample = 10,
            ):
        postpred = self.generate_posterior_predictive_hypercube(n_per_sample)
        new_dim  = np.setdiff1d(np.arange(self.nCol), given_dims)
        w_new    = np.linspace(1,400,1000)
        w_given  = postpred.T[given_dims].mean(axis = 1) / (1 - given_vec_quantile)
        min_thus_far = (postpred.T[given_dims].T / w_given).min(axis = 1)
        args = zip(
            w_new.reshape(-1,1),
            repeat(postpred.T[new_dim].reshape(-1,1)),
            repeat(min_thus_far),
            )
        out  = np.array(list(map(condsurv_at_w_precomp, args)))
        return np.hstack((w_new.reshape(-1,1), out.reshape(-1,1) / min_thus_far.mean()))

    def condsurv_at_quantile_std(
            self,
            target_dims : np.ndarray,
            given_dims : np.ndarray,
            given_vec_quantile : np.ndarray,
            prediction_range = (0.001, 8, 500),
            n_per_sample = 10,
            obs = None
            ):
        assert given_dims.shape[0] == given_vec_quantile.shape[0]
        print('Target {}, Given {}, storm {}'.format(target_dims, given_dims, obs))
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
        # if (target_dims.shape[0] == 1):
        #     pred_bounds = np.linspace(*prediction_range, 500)
        # elif (target_dims.shape[0] > 1):
        #     pred_bounds = np.linspace(*prediction_range, 500)
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

        nsplit = target_z.shape[0] // 200 # arrays of size 200 per

        target_z_list = np.array_split(target_z, nsplit)
        args = zip(
            target_z_list, 
            repeat(postpred.T[target_dims].T), 
            repeat(min_thus_far),
            )
        # out = []
        # for _ in range(len(target_z_list)):
        #     out.append(condsurv_helper(next(args)))
        with mp.Pool(processes = mp.cpu_count(), initializer = limit_cpu) as pool:
            out = np.vstack(list(pool.map(condsurv_helper, args)))
        
        out.T[-1] /= min_thus_far.mean()
        return out
    
    def condsurv_2d_at_quantile_std(
            self,
            given_dims,
            given_vec_quantile,
            n_per_sample = 10,
            ):
        postpred = self.generate_posterior_predictive_hypercube(n_per_sample)
        new_dims = np.setdiff1d(np.arange(self.nCol), given_dims)
        w_given = postpred.T[given_dims].mean(axis = 1) / (1 - given_vec_quantile)
        w_new   = np.linspace(1,400,1000)
        xx, yy  = np.meshgrid(w_new, w_new)
        w_new_grid = np.array((xx.ravel(),yy.ravel())).T
        min_thus_far = (postpred.T[given_dims].T / w_given).min(axis = 1)
        args = zip(w_new_grid, repeat(postpred.T[new_dims].T), repeat(min_thus_far))
        out = np.array(list(map(condsurv_at_w_precomp, args)))
        return np.hstack((w_new_grid, out.reshape(-1,1) / min_thus_far.mean()))

    def condsurv_1d_at_quantile_real(
            self,
            given_dims, given_vec_quantile, prediction_bounds = (0.8, 1.),
            n_per_sample = 10,
            ):
        """ Returns conditional survival curve with real valued margins in 1d """
        # descale_pareto(Z,P):  raw = P[0] + P[1] * (Z**P[2] - 1) / P[2]
        new_dim = np.setdiff1d(np.arange(self.nCol), given_dims)[0]
        surv = self.condsurv_1d_at_quantile_std(
                given_dims, given_vec_quantile, prediction_bounds, n_per_sample
                )
        surv.T[0] = descale_pareto(surv.T[0], self.data.P.T[new_dim])
        return surv

    def condsurv_2d_at_quantile_real(
            self,
            given_dims, given_vec_quantile, prediction_bounds = (0.8, 1.),
            n_per_sample = 10,
            ):
        """ Returns conditional survival curve with real valued margins in 2d """
        new_dims = np.setdiff1d(np.arange(self.nCol), given_dims)
        surv = self.condsurv_2d_at_quantile_std(
                given_dims, given_vec_quantile, prediction_bounds, n_per_sample
                )
        surv.T[0] = descale_pareto(surv.T[0], self.data.P.T[new_dims[0]])
        surv.T[1] = descale_pareto(surv.T[1], self.data.P.T[new_dims[1]])
        return surv

    def load_raw(self, path = '', raw = None, *args, **kwargs):
        data = self.data # copy existing
        if os.path.exists(path):
            raw = pd.read_csv(path)
        elif type(raw) is np.ndarray:
            pass
        else:
            raise TypeError('Pass either path or raw!')
        self.data = Data_From_Raw(raw, *args, **kwargs)
        try:
            self.data.X = data.X
        except AttributeError:
            pass
        return

    pass

def ResultFactory(model_type, fitted_path, raw, raw_args = {}):
    class Result(Results[model_type], Conditional_Survival):
        pass

    result = Result(fitted_path)
    if type(raw) is np.ndarray:
        result.load_raw(raw = raw, **raw_args)
    elif type(raw) is str:
        if os.path.exists(raw):
            result.load_raw(path = raw, **raw_args)
        else:
            raise ValueError('Path string must point at file')
    else:
        raise TypeError('raw should be string or !')

    return result

# Scenario of interest
# Mouth of Bay:
#  0 : Beebe Hospital
#  1 : Henlopen Memorial Park
#  2 : Paramount Air Airport
#  3 : Cape Regional Medical Center
# lower bay:
#   0: Beebe Hospital
#   1: Henlopen Memorial Park 
#   6: Smyrna Airport
# Upper Bay:
#   5: Egg Island Fish and Wildlife Refuge
#   7: Salem Airfield
# Back of Bay
#   7: Salem Airfield
#   8: Rainbow's End Airport
#  Back of Bay; w/ Penn:
#   11: Penn's Landing
# Locations of Interest:
#   4: Dover AFB
#   9: PIA
#   10: Packer Avenue Terminal

if __name__ == '__main__':
    if False:
        fitted_path = "./output/dppprg/results_2_1e-1.db"
        raw_path    = "./datasets/ivt_nov_mar.csv"
        model_type = 'dppprg'
        r = ResultFactory(model_type, fitted_path, raw_path)

        if not os.path.exists('./condsurv'):
            os.mkdir('./condsurv')
        rd1_r = [
            r.condsurv_1d_at_quantile_real(np.setdiff1d(np.arange(8), np.array(i, dtype = int)), 0.9)
            for i in range(8)
            ]
        rd1_df_r = [pd.DataFrame(rd1i) for rd1i in rd1_r]
        for i, df in enumerate(rd1_df_r):
            df['column'] = i
        rd1_s = [
            r.condsurv_1d_at_quantile_std(np.setdiff1d(np.arange(8), np.array(i, dtype = int)), 0.9)
            for i in range(8)
            ]
        rd1_df_s = [pd.DataFrame(rd1i) for rd1i in rd1_s]
        for i, df in enumerate(rd1_df_s):
            df['column'] = i

        d2s = [np.array((i,j), dtype = int) for i in range(8) for j in range(8) if j > i]
        rd2_r = [r.condsurv_2d_at_quantile_real(np.setdiff1d(np.arange(8), cs), 0.9) for cs in d2s]
        rd2_df_r = [pd.DataFrame(rd2i) for rd2i in rd2_r]
        for cols, df in zip(d2s, rd2_df_r):
            df['Column1'] = cols[0]
            df['Column2'] = cols[1]
        rd2_s = [r.condsurv_2d_at_quantile_std(np.setdiff1d(np.arange(8), cs), 0.9) for cs in d2s]
        rd2_df_s = [pd.DataFrame(rd2i) for rd2i in rd2_s]
        for cols, df in zip(d2s, rd2_df_s):
            df['Column1'] = cols[0]
            df['Column2'] = cols[1]

        rd1_df_r_ = pd.concat(rd1_df_r)
        rd1_df_s_ = pd.concat(rd1_df_s)
        rd2_df_r_ = pd.concat(rd2_df_r)
        rd2_df_s_ = pd.concat(rd2_df_s)

        out_path_1d_r = './condsurv/condsurv_ivt8_1d_r.csv'
        out_path_1d_s = './condsurv/condsurv_ivt8_1d_s.csv'
        out_path_2d_r = './condsurv/condsurv_ivt8_2d_r.csv'
        out_path_2d_s = './condsurv/condsurv_ivt8_2d_s.csv'

        rd1_df_r_.to_csv(out_path_1d_r, index = False)
        rd1_df_s_.to_csv(out_path_1d_s, index = False)
        rd2_df_r_.to_csv(out_path_2d_r, index = False)
        rd2_df_s_.to_csv(out_path_2d_s, index = False)
    
    if True:
        scenarios = {
            'Mouth' : np.array([0,1,2,3], int),
            'LowerBay' : np.array([0,1,6], int),
            'UpperBay' : np.array([5,7], int),
            'BackBay1' : np.array([7,8], int),
            'BackBay2' : np.array([7,8,11], int),
            'FullBay' : np.array([0,1,2,3,5,6,7], int),
            }
        scenarios_qtl = {
            key : np.array([0.9] * scenarios[key].shape[0], float)
            for key in scenarios.keys()
            }
        locations_1d = {
            'DoverAFB'  : np.array([4], int),
            'PIA'       : np.array([9], int),
            'PackerAve' : np.array([10], int),
            }
        locations_2d = {
            'DoverAFBPIA'    : np.array([4,9], int),
            'DoverPackerAve' : np.array([4,10], int),
            'PIAPackerAve'   : np.array([9,10], int),
            }
        
        condsurv_scenarios_1d = {
            (lkey, skey) : (
                locations_1d[lkey], scenarios[skey], scenarios_qtl[skey],
                )
            for lkey in locations_1d.keys()
            for skey in scenarios.keys()
            }
        condsurv_scenarios_2d = {
            (lkey, skey) : (
                locations_2d[lkey], scenarios[skey], scenarios_qtl[skey],
                )
            for lkey in locations_2d.keys()
            for skey in scenarios.keys()
            }
        raw_path  = './datasets/slosh/slosh_crt_data.csv.gz'
        slosh     = pd.read_csv(raw_path, compression = 'gzip')
        slosh_ids = slosh.T[:8].T
        slosh_obs = slosh.T[8:].values.astype(np.float64)

        if True:
            pass
            fitted_path = './datasets/slosh/crt/mc_result.pkl'
            model_type  = 'spypprg'
            result      = ResultFactory(
                model_type, fitted_path, raw = slosh_obs, 
                raw_args = {'quantile' : 0.9, 'decluster' : False,},
                )
            nloc        = slosh_ids.shape[0]
            interest    = np.array([4,9,10], dtype = int)

            params = pd.DataFrame(result.data.P.T[[4,9,10]], columns = ('b','a','xi'))
            if not os.path.exists('./condsurv'):
                os.mkdir('./condsurv')
            params['location'] = ('Dover AFB', 'PIA','Packer Ave')
            params.to_csv(
                './condsurv/slosh_crt_params.csv.gz', 
                compression = 'gzip', 
                index = False,
                )
            
            outpath_base = './condsurv/{}_{}.csv.gz'
            for locscen in condsurv_scenarios_1d.keys():
                out = result.condsurv_at_quantile_std(*condsurv_scenarios_1d[locscen])
                pd.DataFrame(out).to_csv(
                    outpath_base.format(*locscen), 
                    index = False, 
                    compression = 'gzip'
                    )
            for locscen in condsurv_scenarios_2d.keys():
                out = result.condsurv_at_quantile_std(*condsurv_scenarios_2d[locscen])
                pd.DataFrame(out).to_csv(
                    outpath_base.format(*locscen), 
                    index = False, 
                    compression = 'gzip'
                    )
        
        if True:
            storm_path  = './datasets/slosh/crt/storms.csv'
            fitted_path = './datasets/slosh/crt/reg_1_result.pkl'
            model_type  = 'spypprgr'
            result      = ResultFactory(
                model_type, fitted_path, raw = slosh_obs, 
                raw_args = {'quantile' : 0.9, 'decluster' : False}, 
                )
            nloc = slosh_ids.shape[0]
            interest = np.array([4, 9, 10], dtype = int)
            if not os.path.exists('./condsurv_reg'):
                os.mkdir('./condsurv_reg')
            params = pd.DataFrame(result.data.P.T[[4,9,10]], columns = ('b','a','xi'))
            params['location'] = ('Dover AFB', 'PIA', 'Packer Ave')
            params.to_csv(
                './condsurv_reg/slosh_crt_params.csv.gz', 
                compression = 'gzip', 
                index = False,
                )
            storms = pd.read_csv(storm_path).values.ravel().tolist()
            # storms = [687,]

            outpath_base = './condsurv_reg/{}_{}_{}_{}.csv.gz'
            for key in condsurv_scenarios_1d.keys():
                for storm in storms:
                    out = result.condsurv_at_quantile_std(*condsurv_scenarios_1d[key], obs = storm)
                    pd.DataFrame(out).to_csv(
                        outpath_base.format(*[*key, '1', storm])
                        )
                out = result.condsurv_at_quantile_std(*condsurv_scenarios_1d[key])
                pd.DataFrame(out).to_csv(
                        outpath_base.format(*[*key, '1', 'base'])
                        )
            for key in condsurv_scenarios_2d.keys():
                for storm in storms:
                    out = result.condsurv_at_quantile_std(*condsurv_scenarios_2d[key], obs = storm)
                    pd.DataFrame(out).to_csv(
                        outpath_base.format(*[*key, '1', storm])
                        )
                    del(out)
                out = result.condsurv_at_quantile_std(*condsurv_scenarios_2d[key])
                pd.DataFrame(out).to_csv(
                        outpath_base.format(*[*key, '1', 'base'])
                        )



# EOF
