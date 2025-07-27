import numpy as np
import numpy.typing as npt
from scipy.special import gammaln

def logp_gammagamma_logshape_summary_old(
        logshapes  : npt.NDArray[np.float64], 
        lys        : npt.NDArray[np.float64], 
        ys         : npt.NDArray[np.float64], 
        n          : npt.NDArray[np.int64]   | int,
        alpha      : npt.NDArray[np.float64] | float, 
        beta       : npt.NDArray[np.float64] | float, 
        xi         : npt.NDArray[np.float64] | float, 
        tau        : npt.NDArray[np.float64] | float,
        validation : bool = True
        ) -> npt.NDArray[np.float64]:
    """
    log-posterior (proportional) for Gamma Gamma Shape parameter
    ---
    shapes : Shape parameter
    lys    : sum(log(y)), per-cluster/dimension
    ys     : sum(y),      per-cluster/dimension
    n      : n(y),        per-cluster
    alpha  : shape hyperprior shape
    beta   : shape hyperprior rate
    xi     : rate hyperprior shape
    tau    : rate hyperprior rate
    ---
    Same function intended to be used for both tempering and non-tempering applications.
    Same function also intended to be used for both hierarchical and non-hierarchical applications
    if non-hierarchical, ensure dimensions match.  If tempering, on prior parameter array,
    insert dummy dimension to ensure dimensions match.
    """
    if len(lys.shape) == 1:
        tempr = False
        hyper = True
    elif len(lys.shape) == 2 and type(alpha) is float:
        tempr = True
        hyper = True
    elif len(lys.shape) == 2 and type(alpha) is np.ndarray:
        tempr = False
        hyper = False
    elif len(lys.shape) == 3:
        tempr = True
        hyper = False
    else:
        raise ValueError("Something don't add up.") 

    if validation:
        assert logshapes.shape == lys.shape
        assert logshapes.shape == ys.shape
        # Check Hierarchical Use
        if any([type(x) is float for x in (alpha, beta, xi, tau)]):
            assert all([type(x) is float for x in (alpha, beta, xi, tau)])
            assert len(lys.shape) in (1,2)
        # Check Non-Hierarchical Use 
        elif any([type(x) is np.ndarray for x in (alpha, beta, xi, tau)]):
            assert all([type(x) is np.ndarray for x in (alpha, beta, xi, tau)])
            assert all([x.shape[-1] == logshapes.shape[-1] for x in (alpha, beta, xi, tau)])
        
    
    # Setup
    out = np.zeros(logshapes.shape)               # (t?, j, d)
    shapes = np.exp(logshapes)                    # (t?, j, d)
    # Calculate Log-Posterior Density
    out += (shapes - 1) * lys                     # (t?, j, d)
    out -= n * gammaln(shapes)                    # (t?, j?, 1)
    out += alpha * logshapes
    out -= beta * shapes
    out += gammaln(n * shapes + xi)
    out -= (n * shapes + xi) * np.log(ys + tau)
    return out

def logp_gamma_gamma_logshape_summary(
        logshapes  : npt.NDArray[np.float64], 
        lys        : npt.NDArray[np.float64], 
        ys         : npt.NDArray[np.float64], 
        n          : npt.NDArray[np.int64],
        alpha      : npt.NDArray[np.float64], 
        beta       : npt.NDArray[np.float64], 
        xi         : npt.NDArray[np.float64], 
        tau        : npt.NDArray[np.float64],
        validation : bool = True
        ) -> npt.NDArray[np.float64]:
    """
    log-posterior (proportional) for Gamma Gamma Shape parameter

    Assuming tempering.  Assume non-hierarchical use.
    That is,
    ---
    logshapes : Shape parameter                      (t,j,d)
    lys       : sum(log(y)), per-cluster/dimension   (t,j,d)
    ys        : sum(y),      per-cluster/dimension   (t,j,d)
    n         : n(y),        per-cluster             (t,j)
    alpha     : shape hyperprior shape               (t,d)
    beta      : shape hyperprior rate                (t,d)
    xi        : rate hyperprior shape                (t,d)
    tau       : rate hyperprior rate                 (t,d)
    ---
    """
    if validation:
        assert logshapes.shape == lys.shape
        assert logshapes.shape == ys.shape
        assert alpha.shape     == beta.shape
        assert xi.shape        == tau.shape
        assert n.shape         == lys.shape[:-1]
        assert logshapes.shape[-1] == alpha.shape[-1] or alpha.shape[-1] == 1
        assert len(logshapes.shape) == 3
        assert len(alpha.shape) == 2
    shapes = np.exp(logshapes)

    out = np.zeros(logshapes.shape)
    out += (shapes - 1) * lys
    out -= n[...,None] * gammaln(shapes)
    out += alpha * logshapes
    out -= beta * shapes
    out += gammaln(n[...,None] * shapes + xi)
    out -= (n[...,None] * shapes + xi) * np.log(ys + tau)
    return out

def logp_resgamma_gamma_logshape_summary_old(
        logshapes  : npt.NDArray[np.float64],
        lys        : npt.NDArray[np.float64],
        ys         : npt.NDArray[np.float64],
        n          : npt.NDArray[np.int64],
        alpha      : npt.NDArray[np.float64],
        beta       : npt.NDArray[np.float64],
        validation : bool = True,
        ) -> npt.NDArray[np.float64]:
    """
    log-posterior (proportional) for Restricted Gamma Gamma shape parameter
    ---
    shapes : Shape parameter
    lys    : sum(log(Y))
    ys     : sum(Y)
    n      : n(Y)
    alpha  : shape hyperprior shape
    beta   : shape hyperprior rate
    """
    if validation:
        assert logshapes.shape == lys.shape
        assert logshapes.shape == ys.shape
        assert logshapes.shape[:-1] == n.shape[:-1]
        if any([type(x) is np.ndarray for x in (alpha, beta)]):
            assert all([type(x) is np.ndarray for x in (alpha, beta)])
            assert alpha.shape[-1] == logshapes.shape[-1]
            assert beta.shape[-1] == logshapes.shape[-1]
            # Check for tempering
            if any([len(x.shape) > 1 for x in (alpha, beta)]):
                assert all([len(x) == 2 for x in (alpha, beta)])
    # Setup
    out = np.zeros(logshapes.shape)
    shapes = np.exp(logshapes)
    # Calculate Log-Posterior Density
    out += (shapes - 1) * lys
    out -= n * gammaln(shapes)
    out += alpha * logshapes
    out -= beta * shapes
    return out 

def logp_resgamma_gamma_logshape_summary(
        logshapes  : npt.NDArray[np.float64], 
        lys        : npt.NDArray[np.float64], 
        ys         : npt.NDArray[np.float64], 
        n          : npt.NDArray[np.int64]   | int,
        alpha      : npt.NDArray[np.float64] | float, 
        beta       : npt.NDArray[np.float64] | float,
        validation : bool = True
        ) -> npt.NDArray[np.float64]:
    """
    log-posterior (proportional) for Gamma Gamma Shape parameter

    Assuming tempering.  Assume non-hierarchical use.
    That is,
    ---
    logshapes : Shape parameter                      (t,j,d)
    lys       : sum(log(y)), per-cluster/dimension   (t,j,d)
    ys        : sum(y),      per-cluster/dimension   (t,j,d)
    n         : n(y),        per-cluster             (t,j)
    alpha     : shape hyperprior shape               (t,d)
    beta      : shape hyperprior rate                (t,d)
    xi        : rate hyperprior shape                (t,d)
    tau       : rate hyperprior rate                 (t,d)
    ---
    """
    if validation:
        assert logshapes.shape == lys.shape
        assert logshapes.shape == ys.shape
        assert alpha.shape     == beta.shape
        assert n.shape         == lys.shape[:-1]
        assert logshapes.shape[-1] == alpha.shape[-1]
        assert len(logshapes.shape) == 3
        assert len(alpha.shape) == 2
    shapes = np.exp(logshapes)

    out = np.zeros(logshapes.shape)
    out += (shapes - 1) * lys
    out -= n[...,None] * gammaln(shapes)
    out += alpha * logshapes
    out -= beta * shapes
    return out

def logd_projgamma_my_mt(
        y     : npt.NDArray[np.float64], 
        shape : npt.NDArray[np.float64], 
        rate  : npt.NDArray[np.float64],
        validation : bool = True,
        ) -> npt.NDArray[np.float64]:
    """
    log-density of Projected Gamma distribution.
    ---
    y     : projected gamma data  (n, d)
    shape : shape parameter       (t, j, d)
    rate  : rate parameter        (t, j, d)
    validation: checks dimensions for compatibility
    ---
    out   : output log-density    # (n, t?, j)
    ----
    If not tempering, translate to same code as if tempering, then reshape.
    """
    if validation:
        assert len(shape.shape) in (2,3)
        assert len(y.shape) == 2
        assert shape.shape == rate.shape
        assert shape.shape[-1] == y.shape[-1]
    if len(shape.shape) == 2:
        shape = shape[None]
        rate  = rate[None]
        tempering = False
    else: 
        tempering = True
    N = y.shape[0]
    T,J,D = shape.shape
    shs = shape.sum(axis = -1)
    out = np.zeros((N,T,J))

    out += np.einsum('tjd,tjd->tj', shape, np.log(rate))
    out -= gammaln(shape).sum(axis = -1)
    out += np.einsum('tjd,nd->ntj', shape - 1, np.log(y))
    out += gammaln(shs)
    out -= shs[None] * np.log(np.einsum('nd,tjd->ntj', y, rate))

    if tempering:
        return out
    else:
        return out.reshape((N,J))

def logd_projresgamma_my_mt(
        y     : npt.NDArray[np.float64], 
        shape : npt.NDArray[np.float64],
        validation : bool = True,
        ) -> npt.NDArray[np.float64]:
    """
    log-density of Projected Restricted Gamma Distribution
    ---
    y     : projected gamma data  # (n, d)
    shape : shape parameter       # (t?, j, d)
    validation: checks dimensions for compatibility
    ---
    out   : output log-density    # (n, t?, j)
    ---
    If not tempering, translate to same code as if tempering, then reshape.
    """
    if validation:
        assert len(shape.shape) in (2,3)
        assert len(y.shape) == 2
        assert shape.shape[-1] == y.shape[-1]
    if len(shape.shape) == 2:
        shape = shape[None] # translate to 3D
        tempering = False   # Set tempering flag
    else:
        tempering = True
    N = y.shape[0]          # Set dimensions
    T,J,D = shape.shape
    shs = shape.sum(axis = -1)
    out = np.zeros((N,T,J))
    out -= gammaln(shape).sum(axis = -1) # (t, j)
    out += np.einsum('tjd,nd->ntj', shape - 1, np.log(y))
    out += gammaln(shs)                  # (t, j)
    out -= (shs[None] * np.log(y.sum(axis = -1))[:,None,None])
    if tempering:
        return out
    else: 
        return out.reshape((N,J))

# EOF