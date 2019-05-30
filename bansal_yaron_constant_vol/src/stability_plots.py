
import matplotlib.pyplot as plt
import numpy as np


def stability_plot(R, 
                   x, y, 
                   xlb, ylb, 
                   dot_loc=None,
                   coords=(-225, 30)):

    text = "Bansal and Yaron constant volatility"

    param1_value, param2_value = dot_loc

    fig, ax = plt.subplots(figsize=(10, 5.7))

    cs1 = ax.contourf(x, y, R.T, alpha=0.5)

    ctr1 = ax.contour(x, y, R.T, levels=[0.0])

    plt.clabel(ctr1, inline=1, fontsize=13)
    plt.colorbar(cs1, ax=ax)

    ax.annotate(text, 
                xy=(param1_value, param2_value),  
                xycoords="data",
                xytext=coords,
                textcoords="offset points",
                fontsize=12,
                arrowprops={"arrowstyle" : "->"})

    ax.plot([param1_value], [param2_value],  "ko", alpha=0.6)

    ax.set_xlabel(xlb, fontsize=16)
    ax.set_ylabel(ylb, fontsize=16)

    plt.savefig("temp.pdf")

    plt.show()

