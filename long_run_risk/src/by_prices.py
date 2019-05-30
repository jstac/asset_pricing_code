"""
Computing price in the Bansal--Yaron model by simulation.  For details on the
model see by_utility.

This code is written for speed, not clarity, and has some repetition.

"""

import numpy as np
from numpy import exp
from numpy.random import randn
from interp2d import lininterp_2d
from numba import jit, njit, prange
from by_utility import default_params

## Build w^* and a global jitted function, constructed using file data
z_grid = np.load('z_grid.npy')
σ_grid = np.load('σ_grid.npy')
w_star = np.load('w_star.npy')

az, bz = min(z_grid), max(z_grid)
aσ, bσ = min(σ_grid), max(σ_grid)
a_vec = np.array([az, aσ])         # lower boundaries
b_vec = np.array([bz, bσ])         # upper boundaries
orders = np.array([len(z_grid), len(σ_grid)])       



def compute_price(params, D_1=1.0, ts_length=100000, z_0=None, σ_0=None, seed=None):
    """
    Compute approximate price P_0 by using Monte Carlo to evaluate

        P_0 = E_0(S_1 D_1 + S_2 D_2 + ...)

    when current state is x = (z, σ)

    """

    pass
    # Logic below needs to be fixed...

    """


    β, γ, ψ, μ_c, ρ, ϕ_z, v, d, ϕ_σ, μ_d, α, ϕ_d = params

    # Set seed
    np.random.seed(seed)

    m_prod = 1
    z, σ = z_0, σ_0

    for t in range(1, ts_length):
        g_c = μ_c + z + σ * randn()
        g_d = μ_d + α * z + ϕ_d * σ * randn()
        # Update state
        z_next = ρ * z + ϕ_z * σ * randn()
        σ2 = v * σ**2 + d + ϕ_σ * randn()
        σ_next = np.sqrt(max(σ2, 0))



    obs = np.empty(num_reps)
    for i in range(num_reps):
        D, S, SD_sum = self.simulate_dividends_and_deflator(D_1=D_1,
                                                            z_0=z, 
                                                            σ_0=σ, 
                                                            seed=i,
                                                            ts_length=ts_length)
        obs[i] = SD_sum

    return obs.mean()


    # Initialize state
    z = 0 if z_0 is None else z_0
    σ = np.sqrt(d / (1 - v)) if σ_0 is None else σ_0

    # Compute price process
    P = np.empty(ts_length)
    Z = np.empty(ts_length)
    Σ = np.empty(ts_length)
    D = np.empty(ts_length+1)
    current_D = 1.0

    for t in range(ts_length):
        # Compute current price
        P[t] = self.compute_price(z, σ, current_D)
        # Update state
        z, σ = self.update_state(z, σ)
        Z[t], Σ[t] = z, σ
        # Update dividend
        D[t+1] = current_D
        g_d = μ_d + α * z + ϕ_d * σ * randn()
        current_D = current_D * np.exp(g_d)

    return P, D, Z, Σ

    D = np.empty(ts_length)
    S = np.empty(ts_length)

    S[1] = M[1]
    D[1] = D_1
    SD_sum = D[1] * S[1]


    for t in range(2, ts_length):
        S[t] = S[t-1] * M[t]
        D[t] = np.exp(g_d[t]) * D[t-1]
        SD_sum += S[t] * D[t]

    return D, S, SD_sum
    """

