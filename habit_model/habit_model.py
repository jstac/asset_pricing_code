"""

The Abel habit model: contrasting stability tests

"""

from scipy.stats import norm
import numpy as np

class AbelModel:
    """
    Represents the model.

    """

    def __init__(self, β=0.99,
                       γ=2.5,
                       ρ=-0.14,
                       σ=0.002,
                       x0=0.05,
                       α=1):

        self.β, self.γ, self.ρ, self.σ = β, γ, ρ, σ
        self.α, self.x0 = α, x0

        # derived constants
        self.b = x0 + σ**2 * (1 - γ)
        self.k0 = β * np.exp(self.b * (1 - γ) + σ**2 * (1 - γ)**2 / 2)
        self.k1 = (ρ - α) * (1 - γ)


    def exponent_analytic(self):

        # Unpack parameters
        β, γ, ρ, σ, x0, α = self.β, self.γ, self.ρ, self.σ, self.x0, self.α 
        b, k0, k1 = self.b, self.k0, self.k1

        s = k1 * b / (1 - ρ)
        t = k1**2 * σ**2 /  (2 * (1 - ρ)**2)
        return np.log(k0) + s + t


    def calin_test(self):
        """
        Computes the test value of Calin et al., in logs.  A return value < 0
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

        return np.log(t1) + t2 + t3 + t4 + t5


