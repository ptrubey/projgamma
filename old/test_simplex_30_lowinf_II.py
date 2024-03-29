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
            GammaPrior(0.05,0.05),
            DirichletPrior(0.25),
            )
    fmix.sample(500000)
    fmix.write_to_disk('./output/fmix/results_30_lowinf_II.db',400000,20)

    res = FMIX_Result('./output/fmix/results_30_lowinf_II.db')
    res.write_posterior_predictive('./output/fmix/postpred_30_lowinf_II.csv')


# EOF
