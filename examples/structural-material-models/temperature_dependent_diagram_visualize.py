#!/usr/bin/env python3
import sys
import numpy as np
import scipy.interpolate as inter
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import xarray as xr
import os.path
import glob
from tqdm import tqdm
import torch
from make_800h_diagram import make_data


g_crit = 0.78


def diagram(gs, A=-2.5324, C=-2.3946, g0=g_crit):
    # B = -0.47
    B = C - A * g0
    
    print("B is :", B)
    yvalues = []
    for g in gs:
        if g <= g0:
            yvalues.append(np.power(10, C))
        else:
            yvalues.append(np.power(10, A * g + B))
    return np.array(yvalues)


if __name__ == "__main__":

    total_erates, total_gs, total_sigmas, total_temps = make_data(eps_dot_0=1e10)
    T_min, T_max = 773.15, 1023.15
    rate_min, rate_max = 1.0e-7, 1.0e-6

    for i, (temp, erate) in enumerate(zip(total_temps, total_erates)):
        # if temp < T_max and temp > T_min:
        if temp >= T_max and erate > rate_max:
            if total_gs[i] <= g_crit:
                plt.semilogy(total_gs[i], total_sigmas[i], "ro")
            else:
                plt.semilogy(total_gs[i], total_sigmas[i], "ro")
        else:
            if total_gs[i] <= g_crit:
                plt.semilogy(total_gs[i], total_sigmas[i], "go")
            else:
                plt.semilogy(total_gs[i], total_sigmas[i], "go")

    plt.semilogy(np.sort(total_gs), diagram(np.sort(total_gs)), "k", lw=3.0)
    plt.semilogy(
        np.array([g_crit] * len(total_gs)), diagram(np.sort(total_gs)), "k--", lw=3.0
    )
    # plt.xlim([0.0, 1.5])
    # plt.ylim([1e-5, 1e-2])
    plt.xlabel("g")
    plt.ylabel("${\sigma}_{f}/{\mu}$")

    plt.legend(
        [
            Line2D([0], [0], color="r", marker="o"),
            # Line2D([0], [0], color="r", marker="o"),
            Line2D([0], [0], color="g", marker="o"),
            # Line2D([0], [0], color="g", marker="o"),
            Line2D([0], [0], color="k", lw=4),
            Patch(facecolor="k", edgecolor=None, alpha=0.5),
        ],
        [
            "T>={}, rate<={}".format(T_max, rate_max),
            # "T>={}, rate<={}".format(T_max, rate_max),
            "T<={} or rate>{}".format(T_max, rate_max),
            # "T<={} or rate>{}".format(T_max, rate_max),
            "bilinear fitting",
        ],
        loc="best",
    )

    plt.grid(True)
    # plt.savefig("800H_Diagram_bilinear.png", dpi = 300)
    plt.show()
    plt.close()
