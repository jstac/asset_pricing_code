"""

Some functions for working with the Mehra Prescott model.  Dividend growth is

    ln D_{t+1} - ln D_t = X_{t+1}

where

    X_{t+1} = ρ X_t + b + σ W_{t+1}

with W_t iid and N(0, 1).  Preferences are CRRA.

"""


import numpy as np
from numpy import sqrt, exp
from scipy.stats import norm

inv_sqrt_2pi = 1 / sqrt(2 * np.pi) 


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


    def sim_state(self, x0=None, num_paths=1000, ts_length=1000):
        """
        Simulate the state process.  If x0 is None, then 
        draw from the stationary distribution.

        """
        ρ, b, σ = self.ρ, self.b, self.σ
        X = np.ones((num_paths, ts_length))
        W = np.random.randn(num_paths, ts_length)

        if x0 is None:
            X[:, 0] = self.smean
        else:
            X[:, 0] = x0

        for t in range(ts_length-1):
            X[:, t+1] = ρ * X[:, t] + b + σ * W[:, t+1]
        return X

        
    def spec_rad_sim(self, num_paths=1000, ts_length=1000):

        β, γ = self.β, self.γ

        X = self.sim_state(num_paths=num_paths, ts_length=ts_length)
        A = β * np.exp((1 - γ) * X)

        A = np.prod(A, axis=1)
        return A.mean()**(1/ts_length)


    def spec_rad_analytic(self):
        # Unpack parameters
        β, γ, ρ, σ = self.β, self.γ, self.ρ, self.σ 
        b = self.b

        k1 = 1 - γ
        s = k1 * b / (1 - ρ)
        t = k1**2 * σ**2 /  (2 * (1 - ρ)**2)
        return β * exp(s + t)


