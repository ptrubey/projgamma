from data import Categorical, Data_From_Raw, Data_From_Sphere, MixedData
from projgamma import GammaPrior
from pandas import read_csv
import numpy as np
from energy import limit_cpu
import models
import os

class Heap(object):
    def __init__(self, kwargs):
        self.__dict__.update(kwargs)
        return

p = Heap({
    'in_path' : './ad/solarflare/data_xv1_is.csv',
    'out_path' : './ad/annthyroid/results_xv1.pkl',
    'model' : 'cdppprgln',
    'outcome' : './ad/annthyroid/outcome_xv1_is.csv',
    'nSamp' : '2000',
    'nKeep' : '1000',
    'nThin' : '10',
    'cats'  : 'None',
    'nMix'  : 30,
    'eta_shape' : '2',
    'eta_rate' : '5e-1',
    'sphere' : 'False',
    'quantile' : 0.95,
    'decluster' : 'False',
    'p' : '10',
    'maxclust' : '300',
    })

if __name__ == '__main__':
    limit_cpu()
    # Verify using a mixed model if using categorical variables
    # -- or not using a mixed model, if not using categorical variables
    if eval(p.cats):
        assert(p.model.startswith('m'))
    else:
        assert(not p.model.startswith('m'))

    Chain  = models.Chains[p.model]
    Result = models.Results[p.model]
    raw    = read_csv(p.in_path).values
    raw    = raw[~np.isnan(raw).any(axis = 1)] # equivalent to na.omit


    ## Initialize Data
    if p.model.startswith('c'):
        data = Categorical(raw)
    elif eval(p.cats):
        if eval(p.sphere):
            data = MixedData(raw, eval(p.cats), eval(p.sphere))
        else:
            try:
                data = MixedData(
                    raw, 
                    eval(p.cats), 
                    decluster = eval(p.decluster), 
                    quantile = float(p.quantile),
                    )
            except:
                data = MixedData(
                    raw, 
                    eval(p.cats), 
                    decluster = eval(p.decluster), 
                    quantile = float(p.quantile),
                    )
    else:
        if eval(p.sphere):
            data = Data_From_Sphere(raw)
        else:
            try:
                data = Data_From_Raw(
                    raw, 
                    decluster = eval(p.decluster), 
                    quantile = float(p.quantile),
                    )
            except:
                data = Data_From_Raw(
                    raw, 
                    decluster = eval(p.decluster), 
                    quantile = float(p.quantile),
                    )

    ## If there's a supplied outcome, initialize it
    if os.path.exists(p.outcome):
        outcome = read_csv(p.outcome).values
        data.fill_outcome(outcome)
    
    ## Initialize Chain
    if p.model[:3] in ('sdp','mdp','cdp'):
        model = Chain(
                    data, 
                    prior_eta = GammaPrior(float(p.eta_shape), float(p.eta_rate)), 
                    p = int(p.p), max_clust_count = int(p.maxclust),
                    )
    elif p.model[:3] in ('sfm','mfm','cfm'):
        model = Chain(data, nMix = int(p.nMix), p = int(p.p),)
    elif p.model[:2] in ('sv', 'mv', 'cv'):
        model = Chain(data, p = int(p.p),)
    else:
        raise ValueError
    
    ## Run Sampler
    model.sample(int(p.nSamp))

    ## Write to disk
    model.write_to_disk(p.out_path, int(p.nKeep), int(p.nThin))
    
# EOF
