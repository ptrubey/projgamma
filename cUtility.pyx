cimport numpy as np
import numpy as np
from scipy.stats import multinomial

cpdef np.ndarray[dtype = np.int_t, ndim = 1] counter(
        np.ndarray[dtype = np.int_t, ndim = 1] delta,
        int stop_point
        ):
    """
    Counts integers that appear in an array up to a stop point.  E.g.,
    an array delta = (1,1,2,3,4,0), with stop point 5 will return (1,2,1,1,1)

    Giving a stop point <= delta.max() is bad juju.
    """
    cdef np.ndarray[dtype = np.int_t, ndim = 1] counts = np.zeros(stop_point, dtype = np.int)
    cdef int iter, ndat
    ndat = delta.shape[0]
    for iter in range(ndat):
        counts[delta[iter]] += 1
    return counts

cpdef int first_zero(np.ndarray[dtype = np.int_t, ndim = 1] arr):
    """ Finds the index of the first zero in an integer array """
    cdef int arr_end = arr.shape[0]
    cdef int i
    for i in range(arr_end):
        if not arr[i]:
            return i
    else:
        return arr_end
    pass

cpdef np.ndarray[dtype = np.int_t, ndim = 1] generate_indices(
        np.ndarray[dtype = np.float_t, ndim = 1] probs,
        int n
        ):
    cdef np.ndarray[dtype = np.int_t, ndim = 1] indices = np.empty(n, dtype = np.int)
    cdef int iter_result, iter_indices
    cdef np.ndarray[dtype = np.int_t, ndim = 1] result = multinomial.rvs(n = n, p = probs)
    iter_indices = 0
    for iter_result in range(result.shape[0]):
        while result[iter_result] > 0:
            indices[iter_indices] = iter_result
            result[iter_result] -= 1
            iter_indices += 1
    return indices

# EOF