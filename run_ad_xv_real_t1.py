import sys, os
import numpy as np
from subprocess import Popen, PIPE, STDOUT

cardio = {
    'source'    : './ad/cardio/data_xv{}_is.csv',
    'outcome'   : './ad/cardio/outcome_xv{}_is.csv',
    'results'   : './ad/cardio/results_xv{}_1.pkl',
    'quantile'  : '0.95',
    'cats'      : '[15,16,17,18,19,20,21,22,23,24]',
    'decluster' : 'False',
    'eta_shape' : '2',
    'eta_rate'  : '1e2',
    'model'     : 'mpypprgln',
    }
cover = {
    'source'    : './ad/cover/data_xv{}_is.csv',
    'outcome'   : './ad/cover/outcome_xv{}_is.csv',
    'results'   : './ad/cover/results_xv{}_1.pkl',
    'quantile'  : '0.998',
    'cats'      : '[9,10,11,12]',
    'decluster' : 'False',
    'eta_shape' : '2',
    'eta_rate'  : '1e2',
    'model'     : 'mpypprgln',
    }
mammography = {
    'source'    : './ad/mammography/data_xv{}_is.csv',
    'outcome'   : './ad/mammography/outcome_xv{}_is.csv',
    'results'   : './ad/mammography/results_xv{}_1.pkl',
    'quantile'  : '0.95',
    'cats'      : '[5,6,7,8]',
    'decluster' : 'False',
    'eta_shape' : '2',
    'eta_rate'  : '1e2',
    'model'     : 'mpypprgln',
    }
pima = {
    'source'    : './ad/pima/data_xv{}_is.csv',
    'outcome'   : './ad/pima/outcome_xv{}_is.csv',
    'results'   : './ad/pima/results_xv{}_1.pkl',
    'quantile'  : '0.90',
    'cats'      : '[7,8,9,10,11,12]',
    'decluster' : 'False',
    'eta_shape' : '2',
    'eta_rate'  : '1e2',
    'model'     : 'mpypprgln',
    }
satellite = {
    'source'    : './ad/satellite/data_xv{}_is.csv',
    'outcome'   : './ad/satellite/outcome_xv{}_is.csv',
    'results'   : './ad/satellite/results_xv{}_1.pkl',
    'quantile'  : '0.95',
    'cats'      : '[36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55]',
    'decluster' : 'False',
    'eta_shape' : '2',
    'eta_rate'  : '1e2',
    'model'     : 'mpypprgln',
    }
annthyroid = {
    'source'    : './ad/annthyroid/data_xv{}_is.csv',
    'outcome'   : './ad/annthyroid/outcome_xv{}_is.csv',
    'results'   : './ad/annthyroid/results_xv{}_1.pkl',
    'quantile'  : '0.95',
    'cats'      : '[6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]',
    'decluster' : 'False',
    'eta_shape' : '2',
    'eta_rate'  : '1e2',
    'model'     : 'mpypprgln',
    }
yeast = {
    'source'    : './ad/yeast/data_xv{}_is.csv',
    'outcome'   : './ad/yeast/outcome_xv{}_is.csv',
    'results'   : './ad/yeast/results_xv{}_1.pkl',
    'quantile'  : '0.90',
    'cats'      : '[4,5]',
    'decluster' : 'False',
    'eta_shape' : '2',
    'eta_rate'  : '1e2',
    'model'     : 'mpypprgln',
    }
## 
solarflare = {
    'source'    : './ad/solarflare/data_xv{}_is.csv',
    'outcome'   : './ad/solarflare/outcome_xv{}_is.csv',
    'results'   : './ad/solarflare/results_xv{}_1.pkl',
    'quantile'  : 'None',
    'cats'      : 'None',
    'decluster' : 'False',
    'eta_shape' : '2',
    'eta_rate'  : '1e2',
    'model'     : 'cdppprgln',
    }


datasets = [cardio, cover, mammography, pima, satellite, annthyroid, yeast]
# datasets = [solarflare]
stepping = '1.1'
ntemps = '1'

if __name__ == '__main__':
    processes = []
    process_args = []

    for dataset in datasets:
        for xv_ in range(5):
            xv = xv_ + 1
            args = [
                sys.executable, 
                'test_generic.py', 
                dataset['source'].format(xv),
                dataset['results'].format(xv),
                dataset['model'],
                '--outcome', dataset['outcome'].format(xv),
                '--cats', dataset['cats'],
                '--quantile', dataset['quantile'],
                '--nSamp', '30000', '--nKeep', '20000', '--nThin', '20',
                '--eta_shape', dataset['eta_shape'],
                '--eta_rate', dataset['eta_rate'],
                '--decluster', dataset['decluster'],
                '--ntemps', ntemps, 
                '--stepping', stepping,
                ]
            process_args.append(args)
            processes.append(Popen(args))

    for process in processes:
        process.wait()
    
    error_proc_ids = np.where(
        np.array([process.returncode for process in processes]) != 0
        )[0]
    
    for error_proc_id in error_proc_ids:
        print(process_args[error_proc_id])

# EOF