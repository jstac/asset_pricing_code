"""

Some functions for working with the Mehra Prescott model.  Dividend growth is

    ln D_{t+1} - ln D_t = X_{t+1}

where

    X_{t+1} = ρ X_t + b + σ W_{t+1}

with W_t iid and N(0, 1).  Preferences are CRRA.

"""


import numpy as np
from numpy import sqrt, exp
from numpy.random import randn
from numba import njit, prange


class MehraPrescott:
    """
    Represents the model.

    """

    def __init__(self, β=0.99,
                       γ=2.5,
                       ρ=0.941,   
                       σ=0.000425,         # Conditional volatility
                       b=0.00104):       # Conditional mean

        self.β, self.γ, self.ρ, self.σ, self.b = β, γ, ρ, σ, b

        # Parameters in the stationary distribution
        self.svar = σ**2 / (1 - ρ**2)
        self.ssd = np.sqrt(self.svar)
        self.smean = self.b / (1 - ρ)



        
def stability_exponent_mc_factory(mp, m=1000, n=1000, parallel_flag=True):
    """
    Compute the stability coefficient by Monte Carlo.

    * mc is an instance of MehraPrescott

    Below, Y_j = exp((1 - γ) sum_{i=1}^n X_i^(j)) where j indexes one time
    series path.

    The return value is 

        ln β + (1/n) * ln (Y.mean())

    """

    ρ, b, σ = mp.ρ, mp.b, mp.σ
    β, γ = mp.β, mp.γ

    smean, ssd = mp.smean, mp.ssd

    @njit(parallel=parallel_flag)
    def stability_exponent_mc(m=1000, n=1000):

        Y_sum = 0.0

        for j in prange(m):
            X_sum = 0.0
            X = smean + ssd * randn()

            for i in range(n):
                X_sum += X
                X = ρ * X + b + σ * randn()

            Y_sum += np.exp((1 - γ) * X_sum)

        Y_mean = Y_sum / m
        return np.log(β) +  np.log(Y_mean) / n

    return stability_exponent_mc


def spec_rad_analytic(mp):
    """
    Compute spec rad by numerical linear algebra.

    * mc is an instance of MehraPrescott

    """
    # Unpack parameters
    β, γ, ρ, σ = mp.β, mp.γ, mp.ρ, mp.σ 
    b = mp.b

    k1 = 1 - γ
    s = k1 * b / (1 - ρ)
    t = k1**2 * σ**2 /  (2 * (1 - ρ)**2)
    return β * exp(s + t)


