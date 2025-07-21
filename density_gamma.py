import numpy as np
import numpy.typing as npt
from scipy.special import gammaln

def logp_gammagamma_logshape_summary(
        logshapes  : npt.NDArray[np.float64], 
        lys        : npt.NDArray[np.float64], 
        ys         : npt.NDArray[np.float64], 
        n          : npt.NDArray[np.int64], 
        alpha      : npt.NDArray[np.float64] | float, 
        beta       : npt.NDArray[np.float64] | float, 
        xi         : npt.NDArray[np.float64] | float, 
        tau        : npt.NDArray[np.float64] | float,
        validation : bool = True
        ):
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
    if validation:
        assert logshapes.shape == lys.shape
        assert logshapes.shape == ys.shape
        assert logshapes.shape == n.shape
        if any([type(x) is np.ndarray for x in (alpha, beta, xi, tau)]):
            # Check if hierarchical use
            assert all([type(x) is np.ndarray for x in (alpha, beta, xi, tau)])
            assert alpha.shape[-1] == logshapes.shape[-1]
            assert beta.shape[-1] == logshapes.shape[-1]
            assert xi.shape[-1] == logshapes.shape[-1]
            assert tau.shape[-1] == logshapes.shape[-1]
            # Check for tempering
            if any([len(x) > 1 for x in (alpha, beta, xi, tau)]):
                assert all([len(x) == 3 for x in (alpha, beta, xi, tau, logshapes, lys, ys)])
    # Setup
    out = np.zeros(logshapes.shape)
    shapes = np.exp(logshapes)
    # Calculate Log-Posterior Density
    out += (shapes - 1) * lys
    out -= n * gammaln(shapes)
    out += alpha * logshapes
    out -= beta * shapes
    out += gammaln(n * shapes + xi)
    out -= (n * shapes + xi) * np.log(ys + tau)
    return out

def logp_resgamma_gamma_logshape_summary(
        logshapes  : npt.NDArray[np.float64],
        lys        : npt.NDArray[np.float64],
        ys         : npt.NDArray[np.float64],
        n          : npt.NDArray[np.int64],
        alpha      : npt.NDArray[np.float64],
        beta       : npt.NDArray[np.float64],
        validation : bool = True,
        ):
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
        assert logshapes.shape == n.shape
        if any([type(x) is np.ndarray for x in (alpha, beta)]):
            assert all([type(x) is np.ndarray for x in (alpha, beta)])
            assert alpha.shape[-1] == logshapes.shape[-1]
            assert beta.shape[-1] == logshapes.shape[-1]
            # Check for tempering
            if any([len(x) > 1 for x in (alpha, beta)]):
                assert all([len(x) == 3 for x in (alpha, beta, logshapes, lys, ys)])
    # Setup
    out = np.zeros(logshapes.shape)
    shapes = np.exp(logshapes)
    # Calculate Log-Posterior Density
    out += (shapes - 1) * lys
    out -= n * gammaln(shapes)
    out += alpha * logshapes
    out -= beta * shapes
    return out

# EOF