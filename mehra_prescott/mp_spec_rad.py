"""
Compute the stability exponent via the spectral radius
"""

from mp_model import *
from stability_plots import *

mp_default = MehraPrescott()


G = 6
gamma_vals = np.linspace(1.01, 12, G)
delta_vals = np.linspace(0.02, 0.1, G)

R = np.empty((G, G))
mp = MehraPrescott()

for i, g in enumerate(gamma_vals):
    for j, d in enumerate(delta_vals):
        mp.delta = d
        mp.gamma = g
        R[i, j] = stability_exponent(mp)


stability_plot(R, 
               gamma_vals, 
               delta_vals, 
               "$\\gamma$", 
               "$\\delta$", 
               dot_loc=(mp_default.gamma, mp_default.delta),
               coords=(40, 40))
        


