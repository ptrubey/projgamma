from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import sqlite3 as sql
import pandas as pd
import os
from genpareto import gpd_fit
from numpy.linalg import norm
from math import pi, sqrt, exp
from scipy.special import erf, erfinv
from cdf import ECDF
from __future__ import annotations
from typing import Self

EPS = np.finfo(float).eps
MAX = np.finfo(float).max

def category_matrix(cats : list[int]) -> npt.NDArray[np.bool_]:
    """ 
    Forms a Boolean Category Matrix
        dims = [(# categorical vars), sum(# categories per var)]
    (c, sum(p_i)) 
    """
    if len(cats) == 0:
        return np.array([])
    catvec = np.hstack(list(np.ones(ncat) * i for i, ncat in enumerate(cats)))
    CatMat = (catvec[:, None] == np.arange(len(cats))).T
    return CatMat

# modified to work on n-d arrays
def euclidean_to_simplex(euc : npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """ projects R_+^d to S_1^{d-1} """
    return (euc + EPS) / (euc + EPS).sum(axis = -1)[...,None]

        # self.pool = get_context('spawn').Pool(
def euclidean_to_hypercube(euc : npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """ Project R_+^d to S_{\infty}^{d-1}"""
    V = (euc + EPS) / (euc + EPS).max(axis = -1)[...,None]
    V[V < EPS] = EPS
    return V

def euclidean_to_psphere(euc : npt.NDArray[np.float64], p : int = 10) -> npt.NDArray[np.float64]:
    """ Project R_+^d to S_p^{d-1} """
    Yp = (euc + EPS) / (((euc + EPS)**p).sum(axis = -1)**(1/p))[...,None]
    Yp[Yp < EPS] = EPS
    return Yp

def euclidean_to_catprob(euc : npt.NDArray[np.float64], catmat : npt.NDArray[np.bool_]) -> npt.NDArray[np.float64]:
    """ 
    Projects R_+^d to \prod_c S_1^{d_c -1}

    euc    : (n,s,d) or (s,d)
    catmat : (c,d)
    """
    seuc = (euc @ catmat.T) + EPS
    neuc = np.einsum('...c,cd->...d', seuc, catmat)
    pis = euc / neuc
    return pis

def cluster_max_row_ids(series : npt.NDArray[np.float64]) -> npt.NDArray[np.int32]:
    nDat = series.shape[0]
    lst = []
    clu = np.empty(0, dtype = int)
    for i in range(nDat):
        if series[i] > 1:
            clu = np.append(clu, i)
        else:
            if clu.shape[0] > 0:
                lst.append(clu)
                clu = np.empty(0, dtype = int)
            else:
                pass
    else:
        if clu.shape[0] > 0:
            lst.append(clu)
    max_ids = []
    for cluster in lst:
        max_ids.append(cluster[np.argmax(series[cluster])])
    return np.array(max_ids)
    max_ids = np.empty(0, dtype = int)
    for cluster in lst:
        max_ids = np.append(max_ids, cluster[np.argmax(series[cluster])])
    return max_ids

def compute_uni_gp_parms(
        rv  : npt.NDArray[np.float64], 
        q   : float,
        ) -> npt.NDArray[np.float64]:
    """ Compute Univariate GP parameters given quantile (length 3) """
    b = np.quantile(rv, q)
    a, xi = gpd_fit(rv, b)
    return np.array((b,a,xi))

def compute_gp_parameters_2tail(raw : np.ndarray, q : float):
    """ Computes GP parameters for both tails. """
    P = np.zeros((2,3,raw.shape[1]))
    P[0] = np.apply_along_axis(lambda x: compute_uni_gp_parms(x, q), 0, raw)
    P[1] = np.apply_along_axis(lambda x: compute_uni_gp_parms(x, q), 0, -raw)
    return P

def compute_gp_parameters_1tail(raw : np.ndarray, q : float):
    """ Computes GP parameters only for the upper tail """
    P = np.zeros((1,3,raw.shape[1]))
    P[0] = np.apply_along_axis(lambda x: compute_uni_gp_parms(x, q), 0, raw)
    return P

def rescale_pareto(
        Z : np.ndarray,
        P : np.ndarray,
        C : np.ndarray = None,
        ) -> np.ndarray:
    if C is None:
        C = np.zeros(Z.shape, int)
    # Bounds Checking
    assert Z.shape[1] == P.shape[2]
    assert Z.shape == C.shape
    assert P.shape[0] == 2
    assert P.shape[1] == 3
    # Transformation
    scratch = np.zeros((2,*Z.shape))
    scratch = Z[None]**P[:,[2]]
    scratch -= 1
    scratch /= P[:,[2]]
    scratch *= P[:,[1]]
    scratch += P[:,[0]]
    # Data on real scale
    out = np.where(C == 0, scratch[0], -scratch[1])
    return out

def standardize_pareto_2tail(
        raw : np.ndarray, 
        P : np.ndarray,
        ) -> tuple:
    """ Do the Pareto Scaling for both tails """
    # Bounds Checking
    assert raw.shape[1] == P.shape[2]
    assert P.shape[0] == 2
    assert P.shape[1] == 3
    # Transformation
    scratch = np.zeros((2, *raw.shape))
    scratch[0] += raw
    scratch[1] -= raw
    scratch -= P[:,[0]]
    scratch /= P[:,[1]]
    scratch *= P[:,[2]]
    scratch += 1.
    with np.errstate(invalid = 'ignore'):
        np.log(scratch, out = scratch)
    np.nan_to_num(scratch, copy = False, nan = -np.inf)
    scratch /= P[:,[2]]
    np.exp(scratch, out = scratch)
    # Checking which regime to respect: maximum or minimum
    C = np.argmax(scratch, axis = 0)
    Z = np.max(scratch, axis = 0)
    return Z, C

def standardize_pareto_1tail(
        raw : np.ndarray,
        P   : np.ndarray,
        ) -> np.ndarray:
    """ Do the Pareto scaling for 1 tail """
    assert raw.shape[1] == P.shape[2]
    assert P.shape[0] == 1
    assert P.shape[1] == 3
    scratch = np.zeros(raw.shape)
    scratch += raw
    scratch -= P[0,[0]]
    scratch /= P[0,[1]]
    scratch *= P[0,[2]]
    scratch += 1
    with np.errstate(invalid = 'ignore'):
        np.log(scratch, out = scratch)
    np.nan_to_num(scratch, copy = False, nan = -np.inf)
    scratch /= P[0,[2]]
    np.exp(scratch, out = scratch)
    return scratch

def rank_transform_pareto(
        raw   : np.ndarray, 
        Fhats : list,
        ) -> np.ndarray:
    """ Do Rank-Transform Standard Pareto Scaling """
    # Bounds Checking
    assert len(raw.shape) == 2
    assert raw.shape[1] == len(Fhats)
    # Transformation
    Z = np.array([
        Fhat.stdpareto(x)
        for Fhat, x in zip(Fhats, raw.T)
        ])
    return Z

def rank_invtransform_pareto(
        Z : np.ndarray,
        Fhats : list,
        ) -> np.ndarray:
    # Bounds Checking
    assert len(Z.shape) == 2
    assert Z.shape[1] == len(Fhats)
    # Transformation
    X = np.ndarray(
        Fhat.FhatInv(z)
        for Fhat, z in zip(Fhats, Z.T)
        )
    return X

class DataBase(object):
    raw  = None # Raw Data
    cats = None
    nCat = None
    iCat = None
    dCat = None

    @staticmethod
    def form_idCat(cats : np.ndarray) -> tuple:
        if len(cats) == 0:
            iCat = np.array([], dtype = int)
            dCat = np.array([], dtype = bool)
            return iCat, dCat
        iCat = np.hstack([
            np.ones(cat, dtype = int) * i for i, cat in enumerate(cats)
            ])
        dCat = np.vstack([
            (iCat == i) for i in range(iCat.max() + 1)
            ])
        return iCat, dCat
    
    @classmethod
    def from_dict(cls, d : dict):
        return cls(**d)

    def to_dict(self) -> dict:
        raise NotImplementedError('Overwrite Me!')
    
    @classmethod
    def from_raw(cls, **kwargs):
        raise NotImplementedError('Overwrite Me!')

class Threshold_2Tail(DataBase):
    raw  = None # Raw data (N x D)
    P    = None # Generalized Pareto Parameters (1 (or 2) x 3 x D)
    Z    = None # Standardized Pareto\
    C    = None # 0 -> Upper tail, 1 -> Lower tail
    W    = None # C in one-hot, (N x (2*D))
    cats = None 

    def rescale(self, Z):
        return rescale_pareto(Z, self.P, self.C)

    @classmethod
    def from_raw(cls, raw : np.ndarray, q : float):
        assert len(raw.shape) == 2
        assert type(q) is float
        P = compute_gp_parameters_2tail(raw, q)
        Z, C = standardize_pareto_2tail(raw, P)
        W = np.stack((C == 0,C == 1), dtype = int).reshape(
            Z.shape[0], 2 * Z.shape[1]
            )
        return cls(raw, P, Z, C, W, q)
    
    def to_dict(self) -> dict:
        d = {
            'raw'   : self.raw,
            'P'     : self.P,
            'Z'     : self.Z,
            'C'     : self.C,
            'W'     : self.W,
            'q'     : self.q,
            }
        return d

    def __init__(self,
            raw : np.ndarray, 
            P   : np.ndarray, 
            Z   : np.ndarray, 
            C   : np.ndarray, 
            W   : np.ndarray, 
            q   : float,
            ):
        self.raw = raw
        self.P = P
        self.Z = Z
        self.C = C
        self.W = W
        self.q = q
        self.cats = np.repeat(2, self.Z.shape[1])
        return
    
    pass

class Threshold_1Tail(Threshold_2Tail):
    @classmethod
    def from_raw(cls, raw : np.ndarray, q : float):
        P = compute_gp_parameters_1tail(raw, q)
        Z = standardize_pareto_1tail(raw, P)
        return cls(raw, P, Z, q)
    
    def __init__(
            self, 
            raw : np.ndarray, 
            P   : np.ndarray, 
            Z   : np.ndarray, 
            q   : float, 
            **kwargs
            ):
        self.raw = raw
        self.P   = P
        self.Z   = Z
        self.q   = q
        return
    
    pass

class RankTransform(DataBase):
    # Vestigial
    nCat = 0
    cats = np.array([])
    iCat = np.array([])
    dCat = np.array([])
    # Relevant
    raw   = None
    Z     = None
    Fhats = []

    @classmethod
    def from_raw(cls, raw : np.ndarray):
        Fhats = list(map(ECDF, raw.T))
        Z     = rank_transform_pareto(X)
        return cls(raw, Z, Fhats)

    def std_pareto_transform(self, X : np.ndarray) -> np.ndarray:
        return rank_transform_pareto(X, self.Fhats)
    
    def to_dict(self) -> dict:
        d = {
            'raw'   : self.raw,
            'Z'     : self.Z,
            'Fhats' : self.Fhats,
            }
        return d

    def __init__(self, raw : np.ndarray, Z : np.ndarray, Fhats : list):
        self.raw = raw
        self.Z = Z
        self.Fhats = Fhats
        return

class Spherical(DataBase):
    V = None
    
    @classmethod
    def from_raw(cls, raw : np.ndarray):
        # Bounds-checking
        assert raw.shape[0] > 0
        assert raw.shape[1] > 0
        # Verify on positive orthant
        assert raw.min() >= 0
        # Transform
        V = euclidean_to_hypercube(raw)
        # Instantiate
        return cls(V)
    
    def to_dict(self):
        d = {
            'V' : self.V,
            }
        return d
    
    def __init__(self, V):
        self.V = V
        return

class Multinomial(DataBase):
    cats = None # number of categories per multinomial variable
    nCat = None # total number of categories (sum of Cats)
    iCat = None # Int Vector associating columns with Vars 
    dCat = None # Bool Array associating columns with Vars
    raw  = None # Originating Data
    W    = None # Multinomial Data

    @classmethod
    def from_raw(
            cls, 
            raw  : np.ndarray, 
            cats : np.ndarray,
            ):
        nCat = cats.sum()
        try:
            assert nCat == raw.shape[1]
        except AssertionError:
            print('Total number of categories must equal number of columns')
            raise
        iCat, dCat = cls.form_idCats(cats)
        W    = raw.copy()
        return cls(cats, nCat, iCat, dCat, raw, W)
    
    def to_dict(self) -> dict:
        d = {
            'cats' : self.cats,
            'nCat' : self.nCat,
            'iCat' : self.iCat,
            'dCat' : self.dCat,
            'raw'  : self.raw,
            'W'    : self.W,
            }
        return d

    def __init__(
            self, 
            cats : np.ndarray, 
            nCat : int, 
            iCat : np.ndarray,
            dCat : np.ndarray,
            raw  : np.ndarray, 
            W    : np.ndarray,
            ):
        self.cats = cats
        self.nCat = nCat
        self.iCat = iCat
        self.dCat = dCat
        self.raw  = raw
        self.W    = W
        return

class Categorical(Multinomial):
    vals = None

    @classmethod
    def from_raw(
            cls, 
            raw : np.ndarray, 
            vals : list,
            ):
        assert len(raw.shape) == 2
        # Verify Supplied Values
        if vals is not None:
            assert len(vals) == raw.shape[1]
            for i in range(raw.shape[1]):
                assert len(set(raw.T[i]).difference(set(vals[i]))) == 0
        # If not supplied, make new one based on existing data
        else:
            vals = [np.unique(raw.T[i]) for i in range(raw.shape[1])]

        dummies = []
        cats  = []
        for i in range(raw.shape[1]):
            dummies.append(np.vstack([raw.T[i] == j for j in vals[i]]))
            cats.append(len(vals[i]))
        cats = np.array(cats, dtype = int)
        nCat = cats.sum()
        iCat = np.hstack([
            np.ones(cat, dtype = int) * i for i, cat in enumerate(cats)
            ])
        dCat = np.vstack([
            np.where(iCat == i)[0] for i in range(iCat.max() + 1)
            ])
        W = np.vstack(dummies).T
        return cls(cats, nCat, iCat, dCat, raw, W, vals)
    
    def to_dict(self):
        d = {
            'cats' : self.cats,
            'nCat' : self.nCat,
            'iCat' : self.iCat, 
            'dCat' : self.dCat,
            'raw'  : self.raw,
            'W'    : self.W,
            'vals' : self.vals
            }
        return d

    def __init__(
            self,
            cats : np.ndarray,
            nCat : int,
            iCat : np.ndarray,
            dCat : np.ndarray,
            raw  : np.ndarray, 
            W    : np.ndarray,
            vals : list,
            ):
        self.vals = vals
        super().__init__(cats, nCat, iCat, dCat, raw, W)
        return

    pass 

class Data(DataBase):
    xh1t = None
    xh2t = None
    rank = None
    sphr = None
    cate = None

    dcls = None

    Z = None
    W = None
    V = None
    R = None
    I = None
    
    cats = None
    nCat = None
    iCat = None
    dCat = None
    
    def to_dict(self) -> dict:
        d = dict()
        for key in ['xh1t','xh2t','rank','cate']:
            if self.__dict__[key] is not None:
                d[key] = self.__dict__[key].to_dict()
        for key in ['Z','W','V','R','I','dcls','cats','nCat','iCat','dCat']:
            d[key] = self.__dict__[key]
        return d
    
    @classmethod
    def from_dict(cls, d : dict) -> Self:
        if 'xh1t' in d.keys():
            xh1t = Threshold_1Tail.from_dict(d['xh1t'])
        else:
            xh1t = None
        if 'xh2t' in d.keys():
            xh2t = Threshold_2Tail.from_dict(d['xh2t'])
        else: 
            xh2t = None
        if 'rank' in d.keys():
            rank = RankTransform.from_dict(d['rank'])
        else: 
            rank = None
        if 'sphr' in d.keys():
            sphr = Spherical.from_dict(d['sphr'])
        else:
            sphr = None
        if 'cate' in d.keys():
            cate = Categorical.from_dict(d['cate'])
        else:
            cate = None
        addlargs = dict()
        for key in ['Z','W','V','R','I','dcls','cats','nCat','iCat','dCat']:
            if key in d.keys():
                addlargs[key] = d[key]
        return cls(xh1t = xh1t, xh2t = xh2t, sphr = sphr,
                   rank = rank, cate = cate, **addlargs)
    
    @classmethod
    def from_components(
            cls,
            xh1t : Threshold_1Tail = None, 
            xh2t : Threshold_2Tail = None, 
            rank : RankTransform   = None,
            sphr : Spherical       = None,
            cate : Categorical     = None,
            dcls : bool            = False,
            ):
        inputs = [xh1t, xh2t, sphr, rank, cate]
        filted = [x for x in inputs if x is not None]
        # Input Checking
        assert len(filted) > 0
        #   All data have same length
        assert filted[0].raw.shape[0] > 0
        assert len(np.unique([x.raw.shape[0] for x in filted])) == 1
        # Regime Checking
        if (xh1t is not None) or (xh2t is not None):
            assert (rank is None) and (sphr is None)
            real_regime = 'threshold'
        elif rank is not None:
            assert (xh1t is None) and (xh2t is None) and (sphr is None)
            real_regime = 'rank'
        elif sphr is not None:
            assert (xh1t is None) and (xh2t is None) and (rank is None)
            real_regime = 'sphere'
        else:
            real_regime = 'none'
        # Data Filling
        N = filted[0].raw.shape[0]
        Z = np.empty((N,0), dtype = float)
        W = np.empty((N,0), dtype = int)
        cats = np.empty((0), dtype = int)
        if real_regime == 'threshold':
            if xh1t is not None:
                Z = np.hstack((Z, xh1t.Z))
            if xh2t is not None:
                Z = np.hstack((Z, xh2t.Z))
                W = np.hstack((W, xh2t.W))
                cats = np.hstack((cats, xh2t.cats))
            R = Z.max(axis = 1)
            V = Z / R[:,None]
            V[V < EPS] = EPS
            if dcls:
                I = cluster_max_row_ids(R)
            else:
                I = np.where(R >= 1)[0]
        elif real_regime == 'rank':
            Z = np.hstack((Z, rank.Z))
            R = Z.max(axis = 1)
            V = Z / R[:,None]
            V[V < EPS] = EPS
            I = np.arange(N)
        elif real_regime == 'sphr':
            V = sphr.V
            I = np.arange(N)
        else:
            V = np.empty((N, 0), dtype = float)
            R = np.repeat(1., N)
            I = np.arange(N)
        if cate is not None:
            W = np.hstack((W, cate.W))
            cats = np.hstack((cats, cate.cats))
        
        nCat = cats.sum()
        iCat, dCat = cls.form_idCat(cats)
        
        return cls(xh1t, xh2t, rank, sphr, cate, 
                   Z[I], W[I], V[I], R[I], I, 
                   cats, nCat, iCat, dCat, dcls)
    
    @classmethod
    def from_raw(
            cls, 
            raw         : np.ndarray, 
            xh1t_cols   : np.ndarray = np.array([], dtype = int),
            xh2t_cols   : np.ndarray = np.array([], dtype = int), 
            xhquant     : float      = None,
            dcls        : bool       = None,
            rank_cols   : np.ndarray = np.array([], dtype = int), 
            sphr_cols   : np.ndarray = np.array([], dtype = int), 
            cate_cols   : np.ndarray = np.array([], dtype = int), 
            cate_val    : list       = None,
            ):
        """ data generation from raw, single table.  columns identified using C indexing. """
        # bounds checking:
        assert len(raw.shape) == 2
        assert np.concatenate(
            [xh1t_cols, xh2t_cols, rank_cols, sphr_cols, cate_cols],
            ).max() < raw.shape[1]
        # Parsing
        if len(xh1t_cols) > 0:
            xh1t_raw = raw.T[xh1t_cols].T
        else:
            xh1t_raw = None
        if len(xh2t_cols) > 0:
            xh2t_raw = raw.T[xh2t_cols].T
        else: 
            xh2t_raw = None
        if len(rank_cols) > 0:
            rank_raw = raw.T[rank_cols].T
        else:
            rank_raw = None
        if len(sphr_cols) > 0:
            sphr_raw = raw.T[sphr_cols].T
        else:
            sphr_raw = None
        if len(cate_cols) > 0:
            cate_raw = raw.T[cate_cols].T
        else:
            cate_raw = None
        # Instantiate
        return cls.from_raw_separated(
            xh1t_raw, xh2t_raw, xhquant, dcls, rank_raw, sphr_raw, cate_raw, cate_val,
            )

    @classmethod
    def from_raw_separated(
            cls, 
            xh1t_raw : np.ndarray = None, 
            xh2t_raw : np.ndarray = None, 
            xhquant  : float      = None,
            dcls     : bool       = False,
            rank_raw : np.ndarray = None,
            sphr_raw : np.ndarray = None,
            cate_raw : np.ndarray = None,
            cate_val : list       = None,
            ):
        # Regime Checking
        if (xh1t_raw is not None) or (xh2t_raw is not None):
            assert xhquant is not None
            assert (rank_raw is None) and (sphr_raw is None)
        # Data Parsing
        #   Threshold 1-Tail
        if xh1t_raw is not None:
            xh1t = Threshold_1Tail.from_raw(xh1t_raw, xhquant)
        else:
            xh1t = None
        #   Threshold 2-tail
        if xh2t_raw is not None:
            xh2t = Threshold_2Tail.from_raw(xh2t_raw, xhquant)
        else: 
            xh2t = None
        #   Rank-Transformed
        if rank_raw is not None:
            rank = RankTransform.from_raw(rank_raw)
        else: 
            rank = None
        #   Sphere
        if sphr_raw is not None:
            sphr = Spherical.from_raw(sphr_raw)
        else:
            sphr = None
        #   Categorical
        if cate_raw is not None:
            cate = Categorical.from_raw(cate_raw, cate_val)
        else:
            cate = None
        return cls.from_components(xh1t, xh2t, rank, sphr, cate, dcls)

    def __init__(
            self, 
            xh1t : Threshold_1Tail, 
            xh2t : Threshold_2Tail, 
            rank : RankTransform,
            sphr : Spherical,
            cate : Categorical, 
            Z    : np.ndarray = None, 
            W    : np.ndarray = None, 
            V    : np.ndarray = None, 
            R    : np.ndarray = None, 
            I    : np.ndarray = None,
            cats : np.ndarray = None,
            nCat : int        = None,
            iCat : np.ndarray = None,
            dCat : np.ndarray = None,
            dcls : bool       = None,
            ):
        self.xh1t = xh1t
        self.xh2t = xh2t
        self.rank = rank
        self.sphr = sphr
        self.cate = cate
        self.Z    = Z
        self.W    = W
        self.V    = V
        self.R    = R
        self.I    = I
        self.cats = cats
        self.nCat = nCat
        self.iCat = iCat
        self.dCat = dCat
        self.dcls = dcls
        self.nCol = Z.shape[1]
        self.nDat = self.I.shape[0]
        return

class Projection(object):
    def set_projection(self):
        if self.data.V.shape[1] > 0:
            self.data.Yp = euclidean_to_psphere(self.data.V, self.p)
        else:
            self.data.Yp = np.empty_like(self.data.V)
        return
    pass

if __name__ == '__main__':
    X1 = np.random.normal(loc = 1., size = (500, 5)) # 5
    X2 = np.random.gamma(shape =  5, scale = 0.5, size = (500, 3)) # 3
    X3 = np.vstack((
        np.random.choice(3, size = 500, p = np.array([1,2,3])/6),
        np.random.choice(3, size = 500, p = np.array([3,2,1])/6),
        np.random.choice(3, size = 500, p = np.array([1,4,1])/6),
        )).T # 3
    X = np.hstack((X1,X2,X3))
    data = Data.from_raw(
        raw = X, 
        xhquant = 0.95,
        dcls = False,
        xh2t_cols = np.array([0,1,2,3,4], dtype = int),
        xh1t_cols = np.array([5,6,7], dtype = int),
        cate_cols = np.array([8,9,10], dtype = int),
        ) # instantiate
    d = data.to_dict() # serialize
    dat2 = Data.from_dict(d) # recover from serialized
    assert all(dat2.cats == data.cats)
    assert np.all(dat2.W == data.W)
    assert np.all(dat2.dCat == data.dCat)
    raise

# EOF
