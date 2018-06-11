"""

Classes and jitted functions for working with normalized (i.e., orthonormal)
Hermite basis functions.

The basis functions form an orthonormal basis of L_2(\pi) when

    \pi = N(mu, sig^2)

If H_i(x) is the standard i-th probabilist's Hermite polynomial evaluated at
x, these functions return 

    e_i(x) = h_i( (x - mu) / sig )

where

    h_i(x) := H_i(x) / sqrt(i!).

"""

import numpy as np
import scipy.misc as sc
from numba import jit
import quantecon as qe
from scipy.special import hermitenorm as H  # For tests
from scipy.integrate import fixed_quad
from scipy.stats import norm

class HermitePoly:
    """
    The purpose of this class is to provide fast evaluations of the form

        h_i(x)

    where h_i is the i-th normalized probabilist's Hermite polynomial.
    The evaluations are vectorized.

    The class also provides a function to evaluate the inner product

        \int f(x) h_i(x) \pi(x) dx

    where f is a supplied function and \pi is the standard normal
    distribution.
    """


    def __init__(self, N, mu=0, sig=1):
        """
        Generate data for Hermite polynomial coeffs, polynomials of order 
        0,..., N-1.

        """
        self.mu = mu
        self.sig = sig
        self.N = N
        self.C = np.zeros((N, N))

        h_coefs(self.C, N)
        h_normalize(self.C, N)
        
    def __call__(self, i, x):
        """
        Evaluate h_i(x).  The function is vectorized in both i and x.

        Parameters:
        ----------
        i : int
            The index

        x : scalar or flat numpy array
        """
        # == Normalize x == #
        x = (x - self.mu) / self.sig

        if np.isscalar(x):
            return h_eval(self.C, i, x)

        else:
            x = x.flatten()
            out_vec = np.empty(len(x))
            h_vec_eval(self.C, i, x, out_vec)
            return out_vec

    def eval_all(self, x):
        """
        Evaluate and return the matrix H[j, k] = h_j(x[k]) where x is a one
        dimensional array (or scalar).

        """
        # == Normalize x == #
        x = (x - self.mu) / self.sig

        if np.isscalar(x):
            out_vec = np.empty(self.N)
            h_eval_over_idx(self.C, self.N, x, out_vec)
            return out_vec
        
        else:
            x = x.flatten()
            out_mat = np.empty((self.N, len(x)))
            h_vec_idx_eval(self.C, self.N, x, out_mat)
            return out_mat


    def inner_prod(self, f, i, quad_size=40):
        """
        Compute the inner product
            
            \int f(x) h_i(x) \pi(x) dx 

        where \pi is the standard normal distribution.
        """
        integrand = lambda x: f(x) * self.__call__(i, x) * norm.pdf(x, loc=self.mu, scale=self.sig)
        std_devs = 5 # Integrate out to 5 std deviations in each direction
        a = self.mu - self.sig * std_devs
        b = self.mu + self.sig * std_devs
        return fixed_quad(integrand, a, b, n=quad_size)[0]




class HermiteFunctionApproximator:
    """
    Approximate f on L_2(\pi) where

        \pi = N(\mu, \sigma^2)

    using the Hermite expansion 

        f_N (x) = \sum_{i=0}^{N-1} \inner{f, e_i} e_i(x)

    """

    def __init__(self, f, N, mu=0, sig=1):

        self.f, self.N = f, N
        self.e = HermitePoly(N, mu=mu, sig=sig)
        self.trim_tolerance = 1e-7

        self.coefs = np.empty(self.N)
        for i in range(self.N):
            y = self.e.inner_prod(f, i)
            if np.abs(y) > self.trim_tolerance:
                self.coefs[i] = y
            else:
                self.coefs[i] = 0.0

    def __call__(self, x):

        return np.dot(self.coefs, self.e.eval_all(x))


class HermiteLinearCombination:
    """
    Provide the function

        g(x) = \sum_{i=0}^{N-1} c_i e_i(x)

    where N = len(c).

    """

    def __init__(self, coefs, mu=0, sig=1):
        """

        Parameters
        ----------
        c : array
            A flat numpy ndarray containing the coefficients
        """
        self.coefs, self.mu, self.sig = coefs, mu, sig
        self.N = len(coefs)
        self.e = HermitePoly(N, mu=mu, sig=sig)

    def __call__(self, x):

        return np.dot(self.coefs, self.e.eval_all(x))



@jit(nopython=True)
def h_coefs(C, N):
    """
    Hermite polynomial coeffs for polynomials of order 0, ..., N-1.

    The function modifies C in place.
    """
    C[0, 0] = 1
    C[1, 1] = 1
    for n in range(1, N-1):
        C[n+1, 0] = -n * C[n-1, 0]
        for k in range(1, n+2):
            C[n+1, k] = C[n, k-1] - n * C[n-1, k]

def h_normalize(C, N):
    """
    Normalize coefficients to make them orthogonal.

    The function modifies C in place.
    """
    for i in range(N):
        m = 1.0 / np.sqrt(sc.factorial(i))
        C[i, :] = C[i, :] * m


@jit(nopython=True)
def h_eval(C, n, x):
    """
    Evaluate h_n(x), the n-th normalized hermite polynomial.  It is assumed
    that the coefficients have already been multiplied by sqrt(n!) to make the
    polynomials orthonormal.
    """
    xpow = 1.0
    sm = C[n, 0]
    for i in range(1, n+1):
        xpow = xpow * x
        sm += C[n, i] * xpow
    return sm

@jit(nopython=True)
def h_eval_over_idx(C, n, x, out_vec):
    """
    Evaluate and return the array [h_i(x) for i in 0,..., n-1].
    """
    for i in range(n):
        out_vec[i] = h_eval(C, i, x)

@jit(nopython=True)
def h_vec_eval(C, n, in_vec, out_vec):
    """
    Evaluate out_vec = h_n(in_vec), modifying out_vec in place.  Make sure
    that these vectors are equal length.
    """
    for i in range(in_vec.shape[0]):
        out_vec[i] = h_eval(C, n, in_vec[i])

@jit(nopython=True)
def h_vec_idx_eval(C, n, in_vec, out_mat):
    """
    Evaluate 
    
        out_mat[j, k] = h_j(in_vec[k]), 
        
    modifying out_mat in place.  
    """
    for k in range(in_vec.shape[0]):
        for j in range(n):
            out_mat[j, k] = h_eval(C, j, in_vec[k])

## == Tests == ##

def test_vec_eval():
    n = 10
    x_vec = np.linspace(-1, 1, 10)
    h = HermitePoly(n)

    h0 = [H(i) / np.sqrt(sc.factorial(i)) for i in range(n)]
    return np.allclose(h0[n-1](x_vec), h(n-1, x_vec))

def test_idx_eval():
    n = 10
    x = 2.0
    h = HermitePoly(n)
    h0 = [H(i) / np.sqrt(sc.factorial(i)) for i in range(n)]
    v = np.empty(n)
    for i in range(n):
        v[i] = h0[i](x)
    return np.allclose(v, h.eval_all(x))


def test_inner_prod(x=5):
    m = 10
    h = HermitePoly(m)
    f = lambda x: x + 2 * x**2
    coefs = h.fourier_coefs(f)
    y = np.sum(coefs * h.eval_all(x))
    print(y)
    print(f(x))
    return np.allclose(y, f(x))


def expansion_plot(f=lambda x: 1 + 4 * np.sin(3 * x) - x**2, n=40):
    """
    A visual test not to be included in the test suite.  Plots true function
    and approximation with m-degree expansion.
    """
    fn = HermiteFunctionApproximator(f, n)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    grid_size = 100
    xvec = np.linspace(-4, 4, grid_size)
    ax.plot(xvec, f(xvec), lw=2, label='true')
    ax.plot(xvec, fn(xvec), lw=2, label='approx')
    ax.set_title(r"n = {}".format(n))
    ax.set_ylim(-25, 20)
    ax.legend()
    plt.show()
    return fn

