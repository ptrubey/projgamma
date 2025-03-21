from dp_rprojgamma import DPMPG, ResultDPMPG
from projgamma import GammaPrior
from data import Data_From_Raw
from pandas import read_csv
from random import shuffle


path = './datasets/ivt_nov_mar.csv'


if __name__ == '__main__':
    raw  = read_csv(path)
    # cols = raw.columns.values.tolist()
    # shuffle(cols)
    # raw  = raw.reindex(columns = cols)
    data = Data_From_Raw(raw, True)
    data.write_empirical('./output/dpmrpg/empirical.csv')

    dpmpg = DPMPG(
        data,
        prior_eta = GammaPrior(2.,1e-1)
        )
    dpmpg.sample(20000)
    dpmpg.write_to_disk('./output/dpmrpg/results_2_1e-1.db', 10000, 2)

    res = ResultDPMPG('./output/dpmrpg/results_2_1e-1.db')
    res.write_posterior_predictive('./output/dpmrpg/postpred_2_1e-1.csv')

# EOF
