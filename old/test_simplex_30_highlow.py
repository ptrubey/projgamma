from simplex import *
from data import Data_From_Raw
from pandas import read_csv

path = './datasets/ivt_nov_mar.csv'

if __name__ == '__main__':
    raw = read_csv(path)
    data = Data_From_Raw(raw, True)
    data.write_empirical('./output/fmix/empirical.csv')

    fmix = FMIX_Chain(
            data, 30,
            GammaPrior(2.,1.),
            DirichletPrior(0.5),
            )
    fmix.sample(50000)
    fmix.write_to_disk('./output/fmix/results_30_highlow.db',25000,5)

    res = FMIX_Result('./output/fmix/results_30_highlow.db')
    res.write_posterior_predictive('./output/fmix/postpred_30_highlow.csv')


# EOF
