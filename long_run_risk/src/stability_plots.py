import numpy as np
import matplotlib.pyplot as plt
from utility_solver import compute_recursive_utility
import unicodedata

def stability_plot(ModelClass,
                   param1,        # string
                   p1_min,        # min value for param1
                   p1_max,        # min value for param1
                   param2,        # string 
                   p2_min,        # min value for param2
                   p2_max,        # min value for param2
                   xlabel=None,
                   ylabel=None,
                   w_star_guess=None,
                   coords=(-225, 30),     # relative location of text
                   G=3):                  # grid size for x and y axes

    # Normalize unicode identifiers
    param1 = unicodedata.normalize('NFKC', param1)
    param2 = unicodedata.normalize('NFKC', param2)

    # Allocate arrays, set up parameter grid
    R = np.empty((G, G))

    # Get default parameter vals for param1 and param2
    md = ModelClass()
    param1_value = md.__getattribute__(param1)
    param2_value = md.__getattribute__(param2)

    # Compute utility at default values
    if w_star_guess is not None:
        md.w_star_guess[:] = w_star_guess
    else:
        compute_recursive_utility(md)

    # Set up grid for param1 and param2
    x_vals = np.linspace(p1_min, p1_max, G) 
    y_vals = np.linspace(p2_min, p2_max, G)

    w = np.copy(md.w_star_guess)

    # Loop through parameters computing test coefficient
    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):

            # Create a new instance and take w_star_guess from
            # the last instance.  Set parameters.
            md_previous = md
            md = ModelClass(build_grids=True)
            md.w_star_guess[:] = md_previous.w_star_guess
            md.__setattr__(param1, x)
            md.__setattr__(param2, y)

            if md.utility_params_differ(md_previous):
                compute_recursive_utility(md)

            f = md.stability_exponent_factory(parallel_flag=False)
            R[i, j] = f(n=1000, num_reps=8000)

    # Now the plot
    point_location=(param1_value, param2_value)

    fig, ax = plt.subplots(figsize=(10, 5.7))
    cs1 = ax.contourf(x_vals, y_vals, R.T, alpha=0.5)
    ctr1 = ax.contour(x_vals, y_vals, R.T, levels=[0.0])

    plt.clabel(ctr1, inline=1, fontsize=13)
    plt.colorbar(cs1, ax=ax, format="%.6f")

    if ModelClass.__name__ == 'BY':
        print_name = 'Bansal-Yaron'
    else:
        print_name = 'Schorfheide-Song-Yaron'

    ax.annotate(print_name, 
             xy=point_location,  
             xycoords="data",
             xytext=coords,
             textcoords="offset points",
             fontsize=12,
             arrowprops={"arrowstyle" : "->"})

    ax.plot(*point_location,  "ko", alpha=0.6)

    #title = ""
    #ax.set_title(title)

    if xlabel is None:
        xlabel = param1
    ax.set_xlabel(xlabel, fontsize=16)

    if ylabel is None:
        ylabel = param2
    ax.set_ylabel(ylabel, fontsize=16)

    ax.ticklabel_format(useOffset=False)

    model_type = ModelClass.__name__

    filename = param1 + param2 + "model_type" + "_" + ".pdf"

    plt.savefig("pdfs/" + filename)
    
    plt.show()

