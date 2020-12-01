"""

Some functions for working with the Mehra Prescott model.  Dividend growth is

    D_{t+1} = X_{t+1} D_t

where (X_t) is a two state Markov chain.

"""


import numpy as np
from numpy import sqrt, exp
from numpy.random import randn
from numba import njit, prange
from scipy.linalg import eigvals
import quantecon as qe


class MehraPrescott:
    """
    Represents the model.

    """

    def __init__(self, beta=0.99,
                       gamma=2.5,
                       delta=0.036,   
                       phi=0.43,    
                       mu=0.018):   

        self.beta, self.gamma, self.delta, self.phi, self.mu = beta, gamma, delta, phi, mu


def stability_exponent(mp):

    """
    Compute spec rad by numerical linear algebra.

    """

    # Unpack parameters
    beta, gamma, delta, phi, mu = mp.beta, mp.gamma, mp.delta, mp.phi, mp.mu

    Pi = ((phi, 1-phi),
          (1-phi, phi))

    x_vals = (1 + mu + delta, 1 + mu - delta)

    Pi = np.array(Pi)

    # Build the matrix V(x, y) = beta (1-gamma) y P(x, y)
    V = np.empty_like(Pi)
    for i in range(len(x_vals)):
        for j, y in enumerate(x_vals):
            V[i, j] = beta * y**(1-gamma) * Pi[i, j]

    # Return the log of r(V)
    return np.log(np.max(np.abs(eigvals(V))))
