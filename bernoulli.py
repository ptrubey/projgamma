from fractions import Fraction
from scipy.special import comb
import numpy as np

# memoisation
bernoulli_cache = {}

NITER = 40

def bernoulli(n: float):
    # check input, if it exists, return value
    if n in bernoulli_cache:
        return bernoulli_cache[n]
    # when input is 0 or 1
    if n == 0:
        value = Fraction(1)
    else:  
        value = Fraction(1)
        for k in range(n):
            value -= Fraction(comb(n, k)) * Fraction(bernoulli(k), (n - k + 1))
    # write to cache
    bernoulli_cache[n] = value
    return Fraction(value.numerator, value.denominator)

seq1 = np.array([float(bernoulli(n)) for n in range(41)])[1:]
seq2 = seq1[1::2]
ints = np.arange(1, seq2.shape[0] + 1)

def lgamma(x : np.ndarray, precision = 2):
    out = np.empty(x.shape)
    out += 0.5 * np.log(2 * np.pi)
    out += (x - 0.5) * np.log(x)
    out -= x
    out += (
        seq2[:precision] / 
            (2 * ints[:precision]) /
                (2 * ints[:precision] - 1) / 
                    (x ** (2 * ints[:precision] - 1))
        ).sum(axis = -1)

if __name__ == '__main__':
    from scipy.special import loggamma
    
    def f(x, precision = 5):
        s = np.arange(1,seq2.shape[0] + 1)
        out = 0.
        out += 0.5 * np.log(2 * np.pi)
        out += (x - 0.5) * np.log(x)
        out -= x
        out += (
            seq2[:precision] / 
                (2 * s[:precision] * (2 * s[:precision] - 1)) / 
                    (x**(2 * s[:precision] - 1))
            ).sum()
        return out









