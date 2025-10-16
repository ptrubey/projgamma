# cython: language_level=3, boundscheck=False
import numpy as np
cimport numpy as np
from libc.math cimport sqrt
from cython.parallel import prange 


cpdef double hcdev(
        np.ndarray[dtype = np.float_t, ndim = 1] X,
        np.ndarray[dtype = np.float_t, ndim = 1] Y,
        ):
    cdef:
        int i, starting_face, ending_face
        double s = 0.
    starting_face = X.argmax()
    ending_face   = Y.argmax()
    if ending_face == starting_face:
        for i in range(X.shape[0]):
            if i != ending_face:
                s += (X[i] - Y[i]) * (X[i] - Y[i])
    else:
        for i in range(X.shape[0]):
            if (i == starting_face):
                pass
            elif (i == ending_face):
                s += (X[i] - (2 - Y[starting_face])) * (X[i] - (2 - Y[starting_face]))
            else:
                s += (X[i] - Y[i]) * (X[i] - Y[i])
    return sqrt(s)

cpdef int argmax(double[:] x) nogil:
    cdef int i, m
    cdef double buf = x[0]
    for i in range(x.shape[0]):
        if x[i] > buf:
            m = i
            buf = x[i]
    return m

cpdef double vector_norm(double[:] x, double[:] y) nogil:
    cdef:
        int i
        double s = 0.
    for i in range(x.shape[0]):
        s += (x[i] - y[i]) * (x[i] - y[i])
    return sqrt(s)

cpdef double vector_sum(double[:] x) nogil:
    cdef:
        int i
        double s = 0.
    for i in range(x.shape[0]):
        s += x[i]
    return s

cpdef double hypercube_deviance(double[:] x, double[:] y) nogil:
    cdef:
        int starting_face, ending_face, i
        double s

    starting_face = argmax(x)
    ending_face = argmax(y)

    if starting_face == ending_face: # If they're on the same face, return euclidean norm
        return vector_norm(x, y)

    s = 0.
    for i in range(x.shape[0]):
        if (i == starting_face):
            pass
        elif (i == ending_face):
            s += (x[i] - (2 - y[starting_face])) * (x[i] - (2 - y[starting_face]))
        else:
            s += (x[i] - y[i]) * (x[i] - y[i])

    return sqrt(s)

cpdef double average_pairwise_deviance_hypercube_1(double[:,:] x) nogil:
    cdef:
        int i, j
        int n = x.shape[0]
        double ss = 0.

    for i in range(n):
        for j in range(n):
            ss += hypercube_deviance(x[i], x[j])

    return ss / (n * n)

cpdef double average_pairwise_deviance_euclidean_1(double[:,:] x) nogil:
    cdef:
        int i, j
        int n = x.shape[0]
        double ss = 0.

    for i in range(n):
        for j in range(n):
            ss += vector_norm(x[i], x[j])

    return ss / (n * n)

cpdef double average_pairwise_deviance_euclidean_2(double[:,:] x, double[:] y) nogil:
    cdef:
        int i
        int n = x.shape[0]
        double ss = 0.

    for i in range(n):
        ss += vector_norm(x[i], y)

    return ss / n

cpdef double average_pairwise_deviance_hypercube_2(double[:,:] x, double[:] y) nogil:
    cdef:
        int i
        int n = x.shape[0]
        double ss = 0.

    for i in range(n):
        ss += hypercube_deviance(x[i], y)

    return ss / n

cpdef double energy_score_euclidean(double[:,:,:] predictions, double[:,:] target):
    """
    Compute average energy score (multivariate CRPS).
    Assuming that the indexing of the target matrix is the same
    as the second axis of the predictions matrix.

    ES = \frac{1}{2}\text{E}_F\lVert Y - Y^{\prime}\rVert - \text{E}_F\lVert Y - y\rVert
    """
    cdef:
        int n
        int nDat  = predictions.shape[0]

    # Accuracy/GoF and Precision, Respectively
    cdef double[:] GF = np.empty(nDat)
    cdef double[:] PR = np.empty(nDat)

    for n in prange(nDat, nogil = True, num_threads = 16):
        PR[n] = average_pairwise_deviance_euclidean_1(predictions[n])
        GF[n] = average_pairwise_deviance_euclidean_2(predictions[n], target[n])

    return  0.5 * vector_sum(PR) - vector_sum(GF)

cpdef double energy_score_hypercube(double[:,:,:] predictions, double[:,:] target):
    """
    Compute average energy score (multivariate CRPS).
    Assuming that the indexing of the target matrix is the same
    as the second axis of the predictions matrix.

    ES = \frac{1}{2}\text{E}_F\lVert Y - Y^{\prime}\rVert - \text{E}_F\lVert Y - y\rVert
    """
    cdef:
        int n
        int nDat  = predictions.shape[0]

    # Accuracy/GoF and Precision, Respectively
    cdef double[:] GF = np.empty(nDat)
    cdef double[:] PR = np.empty(nDat)

    for n in prange(nDat, nogil = True, num_threads = 16):
        PR[n] = average_pairwise_deviance_hypercube_1(predictions[n])
        GF[n] = average_pairwise_deviance_hypercube_2(predictions[n], target[n])

    return  0.5 * vector_sum(PR) - vector_sum(GF)

# EOF
