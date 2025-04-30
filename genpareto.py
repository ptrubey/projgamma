import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize
from itertools import repeat

def gpd_fit(data : npt.NDArray[np.float64], threshold : float) -> npt.NDArray[np.float64]:
    " fits the GPD parameters for a given threshold "
    diff = data - threshold
    excess = diff[diff > 0]
    xbar = excess.mean()
    s2   = excess.var()
    # xi0  = -0.5 * (((xbar * xbar) / s2) - 1.)
    xi0 = np.linspace(-1,1,10)
    sc0 = 0.5 * xbar * (((xbar * xbar) / s2) + 1)
    theta_mat = np.array(list(zip(xi0, repeat(sc0))))

    # negative log-likelihood under gen pareto model
    def gpd_neg_log_lik(theta : npt.NDArray[np.float64]) -> float:
        sc, xi = theta
        cond1 = sc <= 0.
        cond2 = (xi <= 0.) and (excess.max() > (-sc / xi))
        if (cond1 or cond2):
            return 1e9
        else:
            y = np.log(1 + (xi * excess) / sc) / xi
            return len(excess) * np.log(sc) + (1 + xi) * y.sum()
    
    # wrapper for minimize function
    def gpd_nll_wrapper(theta : npt.NDArray[np.float64]) -> float:
        fit = minimize(gpd_neg_log_lik, theta, method = 'L-BFGS-B')
        return np.hstack((fit.x, gpd_neg_log_lik(fit.x)))

    # try model fitting at a range of values for xi.
    res = np.array(list(map(gpd_nll_wrapper, theta_mat)))
    # Verify the model actually successfully fit something
    try: 
        assert not np.all(np.abs(res.T[2] - res.T[2].max()) < 1e-9)
    except AssertionError:
        print('Model Fitting Unsuccessful')
    # return those fitted parameters
    return res[res.T[2].argmin(),:2]

# EOF
