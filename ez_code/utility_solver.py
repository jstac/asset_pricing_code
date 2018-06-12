import numpy as np

def compute_recursive_utility(model, 
                              tol=0.001, 
                              max_iter=50000, 
                              verbose=True):
    """
    Solves for the fixed point of T given an instance model of BY or SSY.

    Writes the result to model.w_star_guess.

    """

    T = model.koopmans_operator_factory()

    error = tol + 1
    i = 1
    w_in = np.copy(model.w_star_guess)

    while error > tol and i < max_iter:
        w_out = T(w_in)
        error = np.max(np.abs(w_in - w_out))
        i += 1
        w_in[:] = w_out

    if verbose and i < max_iter:
        print(f"Converged in {i} iterations")
    if i == max_iter:
        print(f"Hit iteration upper bound!")

    model.w_star_guess[:] = w_out


