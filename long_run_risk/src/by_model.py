"""
Computes recursive utility, spectral radius for Bansal--Yaron model

    g_c = μ_c + z + σ η                 # consumption growth, μ_c is μ

    z' = ρ z + ϕ_z σ e'                 # state process, z here is x in BY

    g_d = μ_d + α z + ϕ_d σ η           # div growth, α here is ϕ in BY

    (σ^2)' = v σ^2 + d + ϕ_σ w'         # v, d, ϕ_σ is v_1, σ^2(1-v_1), σ_w

Innovations are IID and N(0, 1). 

See table IV on page 1489 for parameter values.

"""

import numpy as np
from numpy import exp, log
from numpy.random import randn
from numba import jit, njit, prange
from utils import quantile, lininterp_2d

class BY:
    def __init__(self, 
                 β=0.998,
                 γ=10.0,
                 ψ=1.5,
                 μ_c=0.0015,
                 ρ=0.979,
                 ϕ_z=0.044,
                 v=0.987,
                 d=7.9092e-7,
                 ϕ_σ=2.3e-6,
                 μ_d=0.0015,
                 α=3.0,
                 ϕ_d=4.5,
                 z_grid_size=8, 
                 σ_grid_size=8,
                 mc_draw_size=2500,
                 build_grids=True):

        # Unpack all params
        self.β = β
        self.γ = γ
        self.ψ = ψ 
        self.μ_c = μ_c
        self.ρ = ρ
        self.ϕ_z = ϕ_z
        self.v = v
        self.d = d
        self.ϕ_σ = ϕ_σ
        self.μ_d = μ_d
        self.α = α
        self.ϕ_d = ϕ_d 

        self.z_grid_size = z_grid_size 
        self.σ_grid_size = σ_grid_size 
        self.mc_draw_size = mc_draw_size

        # Build grid and draw shocks for Monte Carlo
        if build_grids:
            self.build_grid_and_shocks()

        # An array to store the current guess of w_star
        self.w_star_guess = np.ones((z_grid_size, σ_grid_size))

    def params(self):
        params = self.β, self.γ, self.ψ, \
                 self.μ_c, self.ρ, self.ϕ_z, \
                 self.v, self.d, self.ϕ_σ, \
                 self.μ_d, self.α, self.ϕ_d 
        return params

    def utility_params_differ(self, other):

        p1 = self.β, self.γ, self.ψ, \
                 self.μ_c, self.ρ, self.ϕ_z, \
                 self.v, self.d, self.ϕ_σ
        p2 = other.β, other.γ, other.ψ, \
                 other.μ_c, other.ρ, other.ϕ_z, \
                 other.v, other.d, other.ϕ_σ
        return not np.allclose(p1, p2)

    def build_grid_and_shocks(self, ts_length=100_000, seed=1234):

        # Unpack params
        β, γ, ψ, μ_c, ρ, ϕ_z, v, d, ϕ_σ, μ_d, α, ϕ_d = self.params()
        z_grid_size = self.z_grid_size 
        σ_grid_size = self.σ_grid_size 
        mc_draw_size = self.mc_draw_size

        # Now build and call a jitted function to compute grid / shocks
        @njit
        def build_grid_and_shocks_jitted():

            np.random.seed(seed)
         
            # Allocate memory and intitialize
            z_vec = np.empty(ts_length)
            σ_vec = np.empty(ts_length)
            z_vec[0] = 0.0
            σ_vec[0] = np.sqrt(d / (1 - v))

            # Generate state
            for t in range(ts_length-1):
                # Update state
                z_vec[t+1] = ρ * z_vec[t] + ϕ_z * σ_vec[t] * randn()
                σ2 = v * σ_vec[t]**2 + d + ϕ_σ * randn()
                σ_vec[t+1] = np.sqrt(max(σ2, 0))

            q1, q2 = 0.05, 0.95  # quantiles
            z_min, z_max = quantile(z_vec, q1), quantile(z_vec, q2)
            σ_min, σ_max = quantile(σ_vec, q1), quantile(σ_vec, q2)
            # Build grid along each state axis
            z_grid = np.linspace(z_min, z_max, z_grid_size)
            σ_grid = np.linspace(σ_min, σ_max, σ_grid_size)

            shocks = randn(2, mc_draw_size)

            return z_grid, σ_grid, shocks

        self.z_grid, self.σ_grid, self.shocks = \
                build_grid_and_shocks_jitted()


    def koopmans_operator_factory(self):
        """
        Build a jitted Koopmans operator T for the BY model.

        """

        # Unpack all params
        β, γ, ψ, μ_c, ρ, ϕ_z, v, d, ϕ_σ, μ_d, α, ϕ_d = self.params()
        θ = (1 - γ) / (1 - 1/ψ)
        z_grid = self.z_grid
        σ_grid = self.σ_grid 
        shocks = self.shocks
        num_shocks = shocks.shape[1]
        nz = len(z_grid)
        nσ = len(σ_grid)

        @njit(parallel=True)
        def T(w):
            """ 
            Apply the operator 

                    Tw(x) = 1 + (Kw^θ(x))^(1/θ)

            induced by the Bansal-Yaron model to a function w.

            Uses Monte Carlo for Integration.

            """

            g = w**θ
            Kg = np.empty_like(g)

            # Apply the operator K to g, computing Kg
            for i in prange(nz):
                z = z_grid[i]
                for j in range(nσ):
                    σ = σ_grid[j]
                    mf = exp((1 - γ) * (μ_c + z) + (1 - γ)**2 * σ**2 / 2)
                    g_expec = 0.0
                    for k in range(num_shocks):
                        ε1, ε2 = shocks[:, k]
                        zp = ρ * z + ϕ_z * σ * ε1
                        σ2p = np.maximum(v * σ**2 + d + ϕ_σ * ε2, 0)
                        σp = np.sqrt(σ2p)
                        g_expec += lininterp_2d(z_grid, σ_grid, g, (zp, σp))
                    g_expec = g_expec /  num_shocks

                    Kg[i, j] = mf * g_expec

            return 1.0 + β * Kg**(1/θ)  # Tw

        # Now return the jitted operator
        return T


    def stability_exponent_factory(self, parallel_flag=True):
                         
        # Unpack params and related objects
        β, γ, ψ, μ_c, ρ, ϕ_z, v, d, ϕ_σ, μ_d, α, ϕ_d = self.params()
        θ = (1 - γ) / (1 - 1/ψ)
        z_grid = self.z_grid
        σ_grid = self.σ_grid 
        shocks = self.shocks
        w_star = self.w_star_guess

        z_0 = 0.0
        σ_0 = np.sqrt(self.d / (1 - self.v))

        @njit(parallel=parallel_flag)
        def compute_stability_exponent(n=1000, num_reps=8000, seed=1234):
            """
            Uses fact that

                M_j = β**θ * exp(g_d_j - γ * g_c_j) * (w(x_j) / 
                                (w(x_{j-1}) - 1)**(θ - 1)

            where w is the value function.
            """
            phi_obs = np.empty(num_reps)

            np.random.seed(seed)

            for m in prange(num_reps):

                # Reset accumulator and state to initial conditions
                phi_prod = 1.0
                z, σ = z_0, σ_0

                for t in range(n):
                    # Calculate W_t
                    W = lininterp_2d(z_grid, σ_grid, w_star, (z, σ))
                    # Calculate g^c_{t+1}
                    g_c = μ_c + z + σ * randn()
                    # Calculate g^d_{t+1}
                    g_d = μ_d + α * z + ϕ_d * σ * randn()
                    # Update state to t+1
                    z = ρ * z + ϕ_z * σ * randn()
                    σ2 = v * σ**2 + d + ϕ_σ * randn()
                    σ = np.sqrt(max(σ2, 0))
                    # Calculate W_{t+1}
                    W_next = lininterp_2d(z_grid, σ_grid, w_star, (z, σ))
                    # Calculate M_{t+1} without β**θ 
                    M =  exp(g_d - γ * g_c) * (W_next / (W - 1))**(θ - 1)
                    phi_prod = phi_prod * M 

                phi_obs[m] = phi_prod

            return θ * log(β) + (1/n) * log(np.mean(phi_obs))

        return compute_stability_exponent


