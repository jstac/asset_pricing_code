"""

Schorfheide--Song--Yaron code

"""


import numpy as np
from numpy.random import randn
from numpy import exp
from numba import jit, njit, prange
from utils import quantile, lininterp_3d

class SSY:
    """
    Class for the SSY model, with state process

            g_c = μ_c + z + σ_c η'

            g_d = μ_d + α z + δ σ_c η' + σ_d υ'

            z' = ρ z + sqrt(1 - ρ^2) σ_z e'

            h_z' = ρ_hz h_z + σ_hz u'

            h_c' = ρ_hc h_c + σ_hc w'

            h_d' = ρ_hd h_d + σ_hd w'

            σ_z = ϕ_z σ_bar exp(h_z)

            σ_c = ϕ_c σ_bar exp(h_c)

            σ_d = ϕ_d σ_bar exp(h_d)


        Innovations are IID and N(0, 1).  

        Default values from May 2017 version of Schorfheide, Song and Yaron.
        See p. 28.

    """
        
    def __init__(self, 
                 β=0.999, 
                 γ=8.89, 
                 ψ=1.97,
                 μ_c=0.0016,
                 ρ=0.987,
                 ϕ_z=0.215,
                 σ_bar=0.0032,
                 ϕ_c=1.0,
                 ρ_hz=0.992,
                 σ_hz=np.sqrt(0.0039),
                 ρ_hc=0.991,
                 σ_hc=np.sqrt(0.0096),
                 μ_d=0.001,
                 α=3.65,                     # = ϕ in SSY's notation
                 δ=1.47,                     # = π in SSY's notation
                 ϕ_d=4.54,
                 ρ_hd=0.969,
                 σ_hd=np.sqrt(0.0447),
                 z_grid_size=25,
                 h_z_grid_size=10,
                 h_c_grid_size=10,
                 h_d_grid_size=10,
                 mc_draw_size=6000,
                 build_grids=True):


        # Preferences
        self.β, self.γ, self.ψ = β, γ, ψ
        # Consumption
        self.μ_c, self.ρ, self.ϕ_z = μ_c, ρ, ϕ_z
        self.σ_bar, self.ϕ_c, self.ρ_hz = σ_bar, ϕ_c, ρ_hz
        self.σ_hz, self.ρ_hc, self.σ_hc  = σ_hz, ρ_hc, σ_hc  
        # Dividends
        self.μ_d, self.α, self.δ = μ_d, α, δ 
        self.ϕ_d, self.ρ_hd, self.σ_hd = ϕ_d, ρ_hd, σ_hd

        self.z_grid_size = z_grid_size 
        self.h_z_grid_size = h_z_grid_size 
        self.h_c_grid_size = h_c_grid_size 
        self.h_d_grid_size = h_d_grid_size 
        self.mc_draw_size = mc_draw_size

        # Build grid and draw shocks for Monte Carlo
        if build_grids:
            self.build_grid_and_shocks()

        # An array to store the current guess of w_star
        self.w_star_guess = np.ones((z_grid_size, h_z_grid_size, h_c_grid_size))

    def params(self):
        return self.β, self.γ, self.ψ, \
               self.μ_c, self.ρ, self.ϕ_z, \
               self.σ_bar, self.ϕ_c, self.ρ_hz, \
               self.σ_hz, self.ρ_hc, self.σ_hc,  \
               self.μ_d, self.α, self.δ, \
               self.ϕ_d, self.ρ_hd, self.σ_hd 

    def utility_params_differ(self, other):
        p1 = self.β, self.γ, self.ψ, \
                 self.μ_c, self.ρ, self.ϕ_z, \
                 self.σ_bar, self.ϕ_c, self.ρ_hz, \
                 self.σ_hz, self.ρ_hc, self.σ_hc
        p2 = other.β, other.γ, other.ψ, \
                 other.μ_c, other.ρ, other.ϕ_z, \
                 other.σ_bar, other.ϕ_c, other.ρ_hz, \
                 other.σ_hz, other.ρ_hc, other.σ_hc
        return not np.allclose(p1, p2)

    def simulate_paths(self, n=1_000_000, seed=1234):
        """
        Creates a jitted function to generate and return the state process,
        along with consumption and dividends.  Dates for returned processes
        are

            x[0], ..., x[n-1]
            g_c[0], ..., g_c[n-2]
            g_d[0], ..., g_d[n-2]

        """
        β, γ, ψ, μ_c, ρ, ϕ_z, σ_bar, ϕ_c, ρ_hz, σ_hz, ρ_hc, σ_hc, μ_d, α, δ, \
           ϕ_d, ρ_hd, σ_hd = self.params()

        # Some useful constants
        τ_z = ϕ_z * σ_bar
        τ_c = ϕ_c * σ_bar
        τ_d = ϕ_d * σ_bar
        κ = np.sqrt(1 - ρ**2)

        @njit
        def jitted_sim_paths():

            np.random.seed(seed)

            # Allocate memory for states with initial conditions at the stationary
            # mean, which is zero
            z_vec = np.zeros(n)
            h_z_vec = np.zeros(n)
            h_c_vec = np.zeros(n)
            h_d_vec = np.zeros(n)

            # Allocate memory for consumption and dividend growth
            g_c_vec = np.empty(n-1)
            g_d_vec = np.empty(n-1)

            # Simulate 
            for t in range(n-1):

                # Simplify names of state variables
                z, h_z, h_c, h_d = z_vec[t], h_z_vec[t], h_c_vec[t], h_d_vec[t]
                # Map h to σ
                σ_z = τ_z * exp(h_z)
                σ_c = τ_c * exp(h_c)
                σ_d = τ_d * exp(h_d)
                # Simulate consumption and dividends given state
                g_c_vec[t] = μ_c + z + σ_c * randn()
                g_d_vec[t] = μ_d + α * z + δ * σ_c * randn() + σ_d * randn()
                # Update states
                z_vec[t+1] = ρ * z + κ * σ_z * randn()
                h_z_vec[t+1] = ρ_hz * h_z + σ_hz * randn()
                h_c_vec[t+1] = ρ_hc * h_c + σ_hc * randn()
                h_d_vec[t+1] = ρ_hd * h_d + σ_hd * randn()

            return z_vec, h_z_vec, h_c_vec, h_d_vec, g_c_vec, g_d_vec

        return jitted_sim_paths()


    def build_grid_and_shocks(self, 
                                ts_length=10_000_000, 
                                seed=1234,
                                q1=0.025,  # lower quantile
                                q2=0.925): # upper quantile

        paths = self.simulate_paths()
        z_vec, h_z_vec, h_c_vec, h_d_vec, g_c, g_d = paths

        # Obtain quantiles of state process time series
        z_min, z_max = quantile(z_vec, q1), quantile(z_vec, q2)
        h_z_min, h_z_max = quantile(h_z_vec, q1), quantile(h_z_vec, q2)
        h_c_min, h_c_max = quantile(h_c_vec, q1), quantile(h_c_vec, q2)
        h_d_min, h_d_max = quantile(h_d_vec, q1), quantile(h_d_vec, q2)

        # Build grid along each state axis using these quantiles as bounds
        self.z_grid = np.linspace(z_min, z_max, self.z_grid_size)
        self.h_z_grid = np.linspace(h_z_min, h_z_max, self.h_z_grid_size)
        self.h_c_grid = np.linspace(h_c_min, h_c_max, self.h_c_grid_size)
        self.h_d_grid = np.linspace(h_d_min, h_d_max, self.h_d_grid_size)

        # Shocks for Monte Carlo integration
        np.random.seed(1234)
        self.shocks = randn(3, self.mc_draw_size)


    def koopmans_operator_factory(self):
        """
        Build and return a jitted version of the operator 

            Tw(x) = 1 + (Kw^θ(x))^(1/θ)

        where

            Kg(z, h_z, h_c) 
               =  exp((1-γ)(μ_c + z) + (1-γ)^2 σ_c^2 / 2) E g(z', h_z', h_c')

        Here σ_c = ϕ_c σ_bar exp(h_c) and (z', h_z', h_c') update via the
        dynamics for the SSY model.  
        
        When we write x as the state, the meaning is

            x = (z, h_z, h_c)

        """

        # Unpack parameters and related objects
        β, γ, ψ, μ_c, ρ, ϕ_z, σ_bar, ϕ_c, ρ_hz, σ_hz, ρ_hc, σ_hc, μ_d, α, δ, \
           ϕ_d, ρ_hd, σ_hd = self.params()
        θ = (1 - γ) / (1 - 1/ψ)
        z_grid = self.z_grid
        h_z_grid = self.h_z_grid
        h_c_grid = self.h_c_grid
        shocks = self.shocks 

        # Some useful constants
        τ_z = ϕ_z * σ_bar
        τ_c = ϕ_c * σ_bar
        κ = np.sqrt(1 - ρ**2)
        nz = self.z_grid_size 
        nh_z = self.h_z_grid_size
        nh_c = self.h_c_grid_size
        ns = self.mc_draw_size

        @njit(parallel=True)
        def T(w):
            g = w**θ
            Kg = np.empty_like(g)

            # Apply the operator K to g, computing Kg
            for i in prange(nz):
                z = z_grid[i]
                for k in range(nh_c):
                    h_c = h_c_grid[k]
                    σ_c = τ_c * exp(h_c)
                    mf = exp((1 - γ) * (μ_c + z) + (1 - γ)**2 * σ_c**2 / 2)
                    for j in range(nh_z):
                        h_z = h_z_grid[j]
                        σ_z = τ_z * exp(h_z)
                        g_exp = 0.0
                        for l in range(ns):
                            η, ω, ϵ = shocks[:, l]
                            zp = ρ * z + κ * σ_z * η
                            h_cp = ρ_hc * h_c + σ_hc * ω
                            h_zp = ρ_hz * h_z + σ_hz * ϵ
                            r = zp, h_zp, h_cp
                            g_exp += lininterp_3d(z_grid, h_z_grid, h_c_grid, g, r)
                        Kg[i, j, k] = mf * (g_exp / ns)

            return 1.0 + β * Kg**(1/θ)  # Tw

        return T


    def compute_spec_rad_of_V(self, 
                              z_0=0.0,
                              h_z_0=0.0,
                              h_c_0=0.0,
                              h_d_0=0.0,
                              n=1000, 
                              num_reps=12_000,
                              use_parallel_flag=True):
                         
        # Unpack params and related objects
        β, γ, ψ, μ_c, ρ, ϕ_z, σ_bar, ϕ_c, ρ_hz, σ_hz, ρ_hc, σ_hc, μ_d, α, δ, \
           ϕ_d, ρ_hd, σ_hd = self.params()
        θ = (1 - γ) / (1 - 1/ψ)

        z_grid = self.z_grid
        h_z_grid = self.h_z_grid
        h_c_grid = self.h_c_grid
        h_d_grid = self.h_d_grid
        shocks = self.shocks 

        # Useful constants
        τ_z = ϕ_z * σ_bar
        τ_c = ϕ_c * σ_bar
        τ_d = ϕ_d * σ_bar
        κ = np.sqrt(1 - ρ**2)

        w_star = self.w_star_guess

        @njit(parallel=use_parallel_flag)
        def compute_spec_rad(z_0=z_0, h_z_0=h_z_0, h_c_0=h_c_0, h_d_0=h_d_0): 
            """
            Uses fact that

                M_j = β**θ * exp(g_d_j - γ * g_c_j) * (w(x_j) / 
                                (w(x_{j-1}) - 1)**(θ - 1)

            where w is the value function.
            """
            phi_obs = np.empty(num_reps)

            for m in prange(num_reps):

                # Set seed
                np.random.seed(m)

                # Reset accumulator and state to initial conditions
                phi_prod = 1.0
                z, h_z, h_c, h_d = z_0, h_z_0,  h_c_0, h_d_0

                # Calculate W_0
                r = z, h_z, h_c
                W = lininterp_3d(z_grid, h_z_grid, h_c_grid, w_star, r)

                for t in range(n):
                    # Map h to σ (only h_z required at this step)
                    σ_z = τ_z * exp(h_z)
                    # Update state to t+1
                    z = ρ * z + κ * σ_z * randn()
                    h_z = ρ_hz * h_z + σ_hz * randn()
                    h_c = ρ_hc * h_c + σ_hc * randn()
                    h_d = ρ_hd * h_d + σ_hd * randn()
                    # Map h to σ for the updated states
                    σ_c = τ_c * exp(h_c)
                    σ_d = τ_d * exp(h_d)
                    # Calculate g^c_{t+1}
                    g_c = μ_c + z + σ_c * randn()
                    # Calculate g^d_{t+1}
                    g_d = μ_d + α * z + δ * σ_c * randn() + σ_d * randn()
                    # Calculate W_{t+1}
                    r = z, h_z, h_c
                    W_next = lininterp_3d(z_grid, h_z_grid, h_c_grid, w_star, r)
                    # Calculate M_{t+1} without β**θ 
                    M = exp(g_d - γ * g_c) * (W_next / (W - 1))**(θ - 1)
                    # Update phi_prod and store W_next for following step
                    phi_prod = phi_prod * M 
                    W = W_next

                phi_obs[m] = phi_prod

            return β**θ * np.mean(phi_obs)**(1/n)

        return compute_spec_rad

    def compute_suped_spec_rad(self, n=1, num_reps=8000):

        z_grid = self.z_grid
        h_z_grid = self.h_z_grid
        h_c_grid = self.h_c_grid
        h_d_grid = self.h_d_grid

        nz = self.z_grid_size
        nh_z = self.h_z_grid_size
        nh_c = self.h_c_grid_size
        nh_d = self.h_d_grid_size


        sr = self.compute_spec_rad_of_V(n=n, num_reps=num_reps, use_parallel_flag=False)

        @njit(parallel=True)
        def compute_suped_spec_rad_jitted():
            sup_val = -np.inf
            for i in prange(nz):
                z = z_grid[i]
                for j in range(nh_z):
                    h_z = h_z_grid[j]
                    for k in range(nh_c):
                        h_c = h_c_grid[k]
                        for l in range(nh_d):
                            h_d = h_d_grid[l]
                            s = sr(z_0=z, h_z_0=h_z, h_c_0=h_c, h_d_0=h_d) 
                            sup_val = max(sup_val, s)
            return sup_val

        return compute_suped_spec_rad_jitted
