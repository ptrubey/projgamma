import argparse

def argparser_dp():
    p = argparse.ArgumentParser()
    p.add_argument('nSamp')
    p.add_argument('nKeep')
    p.add_argument('nThin')
    p.add_argument('eta_shape')
    p.add_argument('eta_rate')
    return p.parse_args()

def argparser_fm():
    p = argparse.ArgumentParser()
    p.add_argument('nSamp')
    p.add_argument('nKeep')
    p.add_argument('nThin')
    p.add_argument('nMix')
    return p.parse_args()

def argparser_v():
    p = argparse.ArgumentParser()
    p.add_argument('nSamp')
    p.add_argument('nKeep')
    p.add_argument('nThin')
    return p.parse_args()

def argparser_generic():
    p = argparse.ArgumentParser()
    p.add_argument('in_path')
    p.add_argument('out_path')
    p.add_argument('model')
    p.add_argument('--outcome', default = 'None')
    p.add_argument('--nSamp', default = '50000')
    p.add_argument('--nKeep', default = '40000')
    p.add_argument('--nThin', default = '10')
    p.add_argument('--cats', default = '[]')
    p.add_argument('--nMix', default = 30)
    p.add_argument('--realtype', default = 'threshold')
    p.add_argument('--sphere', default = 'False')
    p.add_argument('--quantile' , default = 0.95)
    p.add_argument('--decluster', default = 'True')
    p.add_argument('--p', default = '10')
    p.add_argument('--maxclust', default = '300')
    p.add_argument('--ntemps', default = '3')
    p.add_argument('--stepping', default = '1.05')
    p.add_argument('--prior_eta', default = '[2e0,1e-1]')
    p.add_argument('--prior_chi', default = '[1e-1,1e0]')
    p.add_argument('--model_radius', default = 'True')
    p.add_argument('--verbose', default = 'False')
    return p.parse_args()

def argparser_simulation():
    p = argparse.ArgumentParser()
    p.add_argument('in_path')
    p.add_argument('model')
    p.add_argument('nSamp')
    p.add_argument('nKeep')
    p.add_argument('nThin')
    p.add_argument('--nMix')
    p.add_argument('--eta_shape', default = '2')
    p.add_argument('--eta_rate', default = '1e-1')
    p.add_argument('--quantile', default = 0.95)
    p.add_argument('--p', default = '10')
    return p.parse_args()

def argparser_ppl():
    p = argparse.ArgumentParser()
    p.add_argument('path')
    return p.parse_args()

def argparser_cs():
    p = argparse.ArgumentParser()
    p.add_argument('path')
    p.add_argument('raw_path')
    return p.parse_args()

def argparser_ad():
    p = argparse.ArgumentParser()
    p.add_argument('model_path')
    p.add_argument('data_path')
    return p.parse_args()

def argparser_varying_p():
    p = argparse.ArgumentParser()
    p.add_argument('in_path')
    p.add_argument('out_path')
    p.add_argument('model')
    p.add_argument('p')
    p.add_argument('--nSamp', default = '50000')
    p.add_argument('--nKeep', default = '20000')
    p.add_argument('--nThin', default = '30')
    p.add_argument('--eta_shape', default = '2')
    p.add_argument('--eta_rate', default = '0.1')
    return p.parse_args()

if __name__ == '__main__':
    p = argparser_generic()
