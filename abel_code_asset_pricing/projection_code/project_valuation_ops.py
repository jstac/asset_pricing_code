"""
Let q(x, x') be the stochastic kernel of the process

    X_{t+1} = rho X_t + b + sigma W_{t+1}

where {W_t} is IID and standard normal. Let pi be the stationary density of
the {X_t} process and let {e_i} be an orthonormal basis of L_2(R, pi).

Let K be a linear operator from L_2(R, pi) to itself.

We wish to compute the projected matrix

    K_ij := \int (K e_i)(x) e_j(x) pi(x) dx

We focus on type I and type II valuation operators.  Type I operators have the
form

    Kf(x) = \int g(x') f(x') q(x, x') dx'

Type II operators have the form

    Kf(x) = g(x) \int f(x') q(x, x') dx'

As shown in the paper, if we use the basis

    e_i(x) = h_i (tau(x))

where h_i is the i-th normalized Hermite polynomial and

    tau(x) = [(1 - rho^2)^{1/2} / sigma] [x - b / (1 - rho)]

then, for type I operators, the matrix K_ij is given by

    K = M D 
    
when 

    D = diag(rho^0, rho^1, ..., rho^n)

and

    M_ij := \int g(x) e_i(x) e_j(x) pi(x) dx.  
    
For type II operators we have 

    K = D M.

"""


import numpy as np
from .hermite_poly import HermitePoly
from scipy.integrate import quad, fixed_quad
from scipy.linalg import solve, eigvals

inv_sqrt_2pi = 1 / np.sqrt(2 * np.pi) 


class AR1:

    def __init__(self, rho, b, sigma):

        self.rho, self.b, self.sigma = rho, b, sigma

        self.svar = sigma**2 / (1 - rho**2)
        self.ssd = np.sqrt(self.svar)
        self.smean = self.b / (1 - rho)

    def tau(self, x):
        a = np.sqrt(1 - self.rho**2) / self.sigma
        return a * (x - self.b / (1 - self.rho))

    def pi(self, x):
        a = np.exp(-(x - self.smean)**2 / (2.0 * self.svar))
        return (inv_sqrt_2pi / self.ssd) * a



def compute_M(ar1, g, n):
    M = np.empty((n, n))
    h = HermitePoly(n)

    for i in range(n):
        for j in range(i, n):
            # Compute M_ij
            def integrand(x):
                tx = ar1.tau(x)
                a = h(j, tx) * h(i, tx)
                return a * g(x) * ar1.pi(x)
            xa, xb = ar1.smean - 10 * ar1.ssd, ar1.smean + 10 * ar1.ssd 
            val, error = fixed_quad(integrand, xa, xb, n=60)
            M[i, j] = val
            M[j, i] = val

    return M


def compute_K(ar1, g, operator_type=1, n=60):
    lambdas = np.array([ar1.rho**i for i in range(n)])
    D = np.diag(lambdas)
    M = compute_M(ar1, g, n)
    if operator_type == 1:
        return M @ D
    else:
        return D @ M
    

# == A test == #

def test_compute_M(n):
    """
    Should produce the identity matrix for any choice of AR1
    process.

    """
    ar1 = AR1(0.9, 1.0, 1.0)
    g = lambda x: 1
    M = compute_M(ar1, g, n)
    print(np.abs(np.round(M, 5)))
