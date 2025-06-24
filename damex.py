import numpy as np
import pandas as pd
from argparser import argparser_damex as argparser
from collections import defaultdict
from models import Results
from data import Data_From_Raw

class DAMEX_Vanilla(object):
    """ Implements the DAMEX algorithm of Goix et al """
    data  = None
    cones = defaultdict(lambda: 1e-8)

    # @staticmethod
    # def rank_transformation_inner(x):
    #     return 1 / (1 - np.array(list(map(lambda y: (y > x).mean(), x))))

    # def rank_transformation(self):
    #     self.data.V_damex = np.apply_along_axis(self.rank_transformation_inner, 0, self.data.Z)
    #     self.data.H_damex = (self.data.V_damex.T / self.data.V_damex.max(axis = 1)).T
    #     return

    def populate_cones(self):
        self.data.C_damex = (self.data > (self.n / self.k * self.eps)).astype(int)
        for row in self.data.C_damex:
            self.cones[tuple(row)] += (1 / self.k) / self.data.nDat
        return

    def scoring(self, newdata):
        """ 
        args:
            newdata : (n x d) real-valued data that has been rank-transformed
        """
        scores = np.empty(newdata.shape[0])
        c_new = (newdata > (self.n / self.k * self.epsilon)).astype(int)
        for i in range(scores.shape[0]):
            scores[i] = self.cones[tuple(c_new[i])]
        return 


    # def scoring(self):
    #     self.scores = np.empty(self.nDat)
    #     for i in range(self.nDat):
    #        self.scores[i] = self.cones[tuple(self.data.C_damex[i])] / self.V_damex[i].max()
    #    return

    def __init__(self, rankdata, epsilon = 0.1, kfac = 0.5):
        self.data = rankdata
        self.n = rankdata.shape[0]
        self.k = self.n ** kfac
        self.eps = epsilon
        # self.rank_transformation()
        self.populate_cones()
        return

    pass

class DAMEX_PostPred(object):
    """ Implements a modified DAMEX algorithm using the posterior predictive distribution
        as the "training" set. """
    data     = None
    postpred = None
    # cones    = defaultdict(int)

    def populate_cones(self, epsilon):
        C_damex = (self.postpred > epsilon).astype(int)
        cones = defaultdict(float)
        for row in C_damex:
            cones[tuple(row)] += 1 / self.postpred.shape[0]
        return cones

    def scoring_raw(self, epsilon = 0.5):
        cone_prob = self.populate_cones(epsilon)
        scores = np.empty(self.data.nDat)
        for i in range(self.nDat):
            scores[i] = cone_prob[tuple(self.data.V[i] > epsilon)] / self.data.R[i]
        return scores

    def scoring_angular(self, epsilon = 0.5):
        cone_prob = self.populate_cones(epsilon)
        scores = np.empty(self.data.nDat)
        for i in range(self.nDat):
            scores[i] = cone_prob[tuple(self.data.V[i]) > epsilon]
        return scores

    def instantiate_data(self, path, decluster = True):
        """ path: raw data path """
        raw = pd.read_csv(path)
        self.data = Data_From_Raw(raw, decluster)
        return
    pass

if __name__ == '__main__':




    pass

# EOF
