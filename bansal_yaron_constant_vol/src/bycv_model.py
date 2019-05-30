"""

Some functions for working with the Bansal -- Yaron constant volatility model.
Consumption growth and dividend growth are given by 

    ln (C_{t+1} / C_t) = μ_c + X_{t+1} + σ_c ε_{t+1}

    ln (D_{t+1} / D_t) = μ_d + X_{t+1} + σ_d ζ_{t+1}

    X_{t+1} = ρ X_t + σ η_{t+1}

with all innovations iid and N(0, 1).  

Preferences are CRRA.  Also,

    Φ_{t+1} = β exp( ln(D_{t+1} / D_t) - γ ln(C_{t+1} / C_t) )

"""

import numpy as np
from numpy import sqrt, exp
from numpy.random import randn
from numba import njit, prange
from scipy.linalg import eigvals
import quantecon as qe


class BYCV:
    """
    Represents the Bansal--Yaron constant volatility model.  Parameters are
    from Bansal and Yaron, 2004, table II.

    """

    def __init__(self, β=0.998,
                       γ=2.5,
                       ρ=0.979,
                       σ=0.00034, 
                       μ_c=0.0015,
                       σ_c=0.0078,
                       σ_d=0.035,    # 4.5 * 0.0078
                       μ_d=0.0015):

        self.ρ, self.σ = ρ, σ   
        self.σ_c, self.μ_c = σ_c, μ_c
        self.σ_d, self.μ_d = σ_d, μ_d  
        self.γ, self.β = γ, β

        
def stability_exponent_mc_factory(by, m=1000, n=1000, seed=1234, parallel_flag=True):
    """

    Compute the stability coefficient by Monte Carlo.

    * by is an instance of BYCV


    """

    ρ, σ = by.ρ, by.σ
    σ_c, μ_c = by.σ_c, by.μ_c
    σ_d, μ_d = by.σ_d, by.μ_d  
    β, γ = by.β, by.γ

    @njit(parallel=parallel_flag)
    def stability_exponent_mc(m=1000, n=1000):

        np.random.seed(seed)

        Phi_prod_sum = 0.0

        for j in prange(m):
            Phi_prod = 1.0
            X = 0.0

            for i in range(n):
                d_growth = μ_d + X + σ_d * randn()
                c_growth = μ_c + X + σ_c * randn()
                Phi_prod = Phi_prod * β * exp(d_growth - γ * c_growth)
                X = ρ * X + σ * randn()

            Phi_prod_sum += Phi_prod

        Phi_prod_mean = Phi_prod_sum / m

        return np.log(Phi_prod_mean) / n

    return stability_exponent_mc


def stability_exp_analytic(by):
    """
    Compute stability exponent exactly.

    * by is an instance of BYCV

    """
    # Unpack parameters
    ρ, σ = by.ρ, by.σ
    σ_c, μ_c = by.σ_c, by.μ_c
    σ_d, μ_d = by.σ_d, by.μ_d  
    β, γ = by.β, by.γ

    a = np.log(β) + μ_d - γ * μ_c 
    b = σ_d**2 + (γ * σ_c)**2
    c =  σ**2 * (1 - γ)**2 /  (1 - ρ)**2

    return  a + (b + c) / 2


def stability_exp_discretized(by, D=10):
    """
    Compute stability exponent by numerical linear algebra.

    * by is an instance of BYCV

    """

    # Unpack parameters
    ρ, σ = by.ρ, by.σ
    σ_c, μ_c = by.σ_c, by.μ_c
    σ_d, μ_d = by.σ_d, by.μ_d  
    β, γ = by.β, by.γ

    # Discretize the state process
    X_mc = qe.rouwenhorst(D, 0.0, σ, ρ)
    x_vals = X_mc.state_values
    P = X_mc.P

    # Build the matrix 
    #
    #   V(x, y) = β * exp(μ_d - γ*μ_c  + (1-γ)*x + (σ_d**2 + (γ*σ_c)**2) / 2) Π(x, y)
    #
    V = np.empty((D, D))
    for i, x in enumerate(x_vals):
        for j, y in enumerate(x_vals):
            a = β * exp(μ_d - γ*μ_c  + (1-γ)*x + (σ_d**2 + (γ*σ_c)**2) / 2)
            V[i, j] = a * P[i, j]

    # Return the log of r(V)
    return np.log(np.max(np.abs(eigvals(V))))
