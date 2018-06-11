"""

Some functions for working with the Abel habit model.

"""

import numpy as np
from numpy import sqrt, exp
from scipy.stats import norm
import quantecon as qe

inv_sqrt_2pi = 1 / sqrt(2 * np.pi) 


class AbelModel:
    """
    Represents the model.

    """

    def __init__(self, β=0.99,
                       γ=2.5,
                       ρ=0.9,
                       σ=0.002,
                       x0=0.1,
                       α=1,
                       grid_size=60):

        self.β, self.γ, self.ρ, self.σ = β, γ, ρ, σ
        self.α, self.x0 = α, x0

        # derived constants
        self.b = x0 + σ**2 * (1 - γ)
        self.k0 = β * exp(self.b * (1 - γ) + σ**2 * (1 - γ)**2 / 2)
        self.k1 = (ρ - α) * (1 - γ)

        # Parameters in the stationary distribution
        self.svar = σ**2 / (1 - ρ**2)
        self.ssd = sqrt(self.svar)
        self.smean = self.b / (1 - ρ)

        # A discrete approximation of the stationary dist 
        std_range, n = 3, 20
        mc = qe.tauchen(0, 1, std_range, n)
        w_vec = mc.state_values
        self.sx_vec = self.smean + self.ssd * w_vec
        self.sp_vec = mc.P[0, :]  # Any row

        # A grid of points for interpolation
        a, b = self.smean + 3 * self.ssd, self.smean - 3 * self.ssd
        self.x_grid = np.linspace(a, b, grid_size)


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


        

    def A(self, g, Ag, std_range=3, shock_state_size=20):
        """
        Apply A to g and return Ag.  The argument g is a vector, which is
        converted to a function by linear interpolation.
        Integration uses Gaussian quadrature.

        """

        # Unpack parameters
        β, γ, ρ, σ, x0, α = self.β, self.γ, self.ρ, self.σ, self.x0, self.α 
        b, k0, k1 = self.b, self.k0, self.k1

        # Extract state and probs for N(0, 1) shocks
        mc = qe.tauchen(0, 1, std_range, shock_state_size)
        w_vec = mc.state_values
        p_vec = mc.P[0, :]  # Any row, all columns

        # Interpolate g and allocate memory for new g
        g_func = lambda x: np.interp(x, self.x_grid, g)

        # Apply the operator K to g, computing Kg and || Kg ||
        for (i, x) in enumerate(self.x_grid):
            mf = k0 * exp(k1 * x)
            Ag[i] = mf * np.dot(g_func(ρ * x + b + w_vec), p_vec)

        # Calculate the norm of Ag
        Ag_func = lambda x: np.interp(x, self.x_grid, Ag)
        r = np.sqrt(np.dot(Ag_func(self.sx_vec)**2, self.sp_vec))

        return r


    def local_spec_rad_iterative(self, tol=1e-7, max_iter=5000):
        """
        Compute the spectral radius of the operator A associated with the Abel
        model self via the local spectral radios
        """
        n = len(self.x_grid)
        g_in = np.ones(n)
        g_out = np.ones(n)

        error = tol + 1
        r = 1
        i = 1

        while error > tol and i < max_iter:
            s = self.A(g_in, g_out)
            new_r = s**(1/i)
            error = abs(new_r - r)
            g_in = np.copy(g_out)
            i += 1
            r = new_r

        print(f"Converged in {i} iterations")
        return r

    def local_spec_rad_simulation(self, num_paths=1000, ts_length=1000):
        X = self.sim_state(num_paths=num_paths, ts_length=ts_length)
        A = self.k0 * np.exp(self.k1 * X)
        A = np.prod(A, axis=1)
        return A.mean()**(1/ts_length)

    def spec_rad_analytic(self):
        # Unpack parameters
        β, γ, ρ, σ, x0, α = self.β, self.γ, self.ρ, self.σ, self.x0, self.α 
        b, k0, k1 = self.b, self.k0, self.k1

        s = k1 * b / (1 - ρ)
        t = k1**2 * σ**2 /  (2 * (1 - ρ)**2)
        return k0 * exp(s + t)


    def calin_test(self):
        """
        Implements the contraction test of Calin et al.  A return value < 1
        indicates contraction.

        """
        # Unpack
        ρ, σ, γ, x0 = self.ρ, self.σ, self.γ, self.x0
        α, k0, k1, b = self.α, self.k0, self.k1, self.b

        # Set up
        phi = norm()
        theta = x0 + σ**2 * (1 + ρ - α) * (1 - γ)

        z = abs(ρ * k1) * σ / (1 - abs(ρ))
        t1 = k0 * (1 + 2 * phi.cdf(z))

        t2 = x0 * k1
        t3 = σ**2 * (1 - γ)**2 * (ρ - α) * (2 + ρ - α) / 2
        t4 = (ρ * k1 * σ)**2 / (2 * (1 - abs(ρ))**2)
        t5 = abs(ρ * k1 * theta) / (1 - abs(ρ))

        return t1 * exp(t2 + t3 + t4 + t5)


    def ell1_test(self):
        """
        Implements the L1 contraction test.  A return value < 1
        indicates contraction.

        """
        # Unpack
        ρ, σ, γ, x0 = self.ρ, self.σ, self.γ, self.x0
        α, k0, k1, b = self.α, self.k0, self.k1, self.b

        t1 = (k1 * b) / (1 - ρ) + (k1 * σ)**2 / (2 * (1 - ρ**2))
        t2 = np.sqrt(2 * np.pi) * σ
        
        return k0 * exp(t1) / t2

