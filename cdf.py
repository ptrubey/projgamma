"""
cdf.py

Functions for computing (and storing) empirical CDF's, 
    computing rank transformations, etc.
"""
import numpy as np
EPS = np.finfo(float).eps

class ECDF(object):
    X = None
    N = None
    eps = None

    def Fhat(self, Xnew):
        return np.searchsorted(self.X, Xnew, side = 'right') / self.N

    def FhatInv(self, Znew):
        Pnew = (1 - 1 / Znew)
        Lnew = (Pnew * self.N).astype(int)
        Lnew[Lnew > (self.N - 1)] = self.N - 1
        return self.X[Lnew]

    def stdpareto(self, Xnew):
        stdpar = 1 / (1 - self.Fhat(Xnew) + self.eps)
        stdpar[stdpar < 1] = 1.
        return stdpar

    def __init__(self, X):
        assert len(X.shape) == 1 # verify univariate input
        self.X = np.sort(X)
        self.N = X.shape[0]
        self.eps = (1 / self.N) * 0.05
        return

if __name__ == '__main__':
    from numpy.random import uniform
    X = uniform(size = 1000000)
    F = ECDF(X)
    print(F.stdpareto(np.array([0.0000001, 0.5, 0.99999999])))

# EOF