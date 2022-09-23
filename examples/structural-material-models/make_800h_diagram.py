#!/usr/bin/env python3
import sys

sys.path.append("../../../..")
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
from pyoptmat import optimize, experiments, temperature, hardening, models, flowrules

data_dir = "database"
full_tensile_dir = "full-tensile"
full_creep_dir = "full-creep"
full_relaxation_dir = "full-relaxation"
full_strain_cyclic_dir = "full-strain-cyclic"
full_stress_cyclic_dir = "full-stress-cyclic"


def interpolate(strain, stress, targets):
    """
    This is just to make sure all our values line up, in case the model
    adaptively integrated or something
    """
    return inter.interp1d(strain, stress)(targets)


# ================================================#
def convert_to_real(p):
    # ================================================#
    bounds = np.array(
        [
            [-10.0, -1.0],  # A
            [-4.0, -1.0],  # B
        ]
    )

    return bounds[:, 0] + (p * (bounds[:, 1] - bounds[:, 0]))


def make_data(eps_dot_0=1e2):
    # fetch data from Sam
    Ndatas = 2000
    nominal_strain = np.linspace(0, 0.2, Ndatas)
    path = ""
    fnames = glob.glob(path + os.path.join(full_tensile_dir, "*.csv"))
    temps = []
    erates = []
    strain_index = []
    yield_strength = []
    for fn in fnames:
        with open(fn, "r") as f:
            metadata = f.readline().strip().split(",")
            temps.append(float(metadata[4]))
            erates.append(float(metadata[5]))
            data = np.loadtxt(f, delimiter=",")
            eng_strain = data[:, 0]
            eng_stress = data[:, 1]
            nominal_stress = interpolate(eng_strain, eng_stress, nominal_strain)
            item_index = np.where(nominal_strain < 0.002)
            stress_index = item_index[0][-1] + 1
            strain_index.append(stress_index)
            yield_strength.append(nominal_stress[stress_index])

            # print(nominal_stress)
    # print(strain_index, yield_strength)
    # print(erates, temps)

    gs = []
    sigmas = []
    for erate, ys, T in zip(erates, yield_strength, temps):
        k = 1.38064e-20
        E_poly = torch.tensor([-3.94833389e-02, -5.20197047e01, 1.95594836e05])
        E = temperature.PolynomialScaling(E_poly)
        nu = torch.tensor(0.31)
        mu = temperature.PolynomialScaling(E_poly / (2 * (1 + nu))).value(
            T=torch.tensor(T)
        )
        a = 0.358e-9
        b = a / 2 * np.sqrt(1**2 + 1**2 + 0**2)
        eps_0 = eps_dot_0
        g = k * T / (mu.item() * 1.0e9 * b**3) * np.log(eps_0 / erate)
        gs.append(g)
        sigmas.append(ys / mu.item())
    """    
    ff = np.poly1d(np.polyfit(gs[:], sigmas[:], deg=1))
    t = np.linspace(0, np.amax(gs[:]), 500)
    plt.plot(gs[:], sigmas[:], 'ko', label='800H Alloy')
    plt.plot(t, ff(t), 'k')
    plt.ylim([1e-4, 1e-2])
    plt.xlabel('g')
    plt.ylabel('${\sigma}_{f}/{\mu}$')
    plt.legend()
    plt.grid(True)
    # plt.savefig("800H_Diagram.png", dpi = 300)
    plt.show()
    plt.close()  
    """
    # fetch data from Roy
    other_temps = [25.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 1000.0]
    other_erates = [1.0e-3] * len(other_temps)
    other_yield_strength = [
        261.0,
        246.0,
        228.0,
        213.0,
        190.0,
        170.0,
        151.0,
        103.0,
        101.0,
        8.0,
    ]

    other_gs = []
    other_sigmas = []
    for erate, ys, T in zip(other_erates, other_yield_strength, other_temps):
        k = 1.38064e-20
        E_poly = torch.tensor([-3.94833389e-02, -5.20197047e01, 1.95594836e05])
        E = temperature.PolynomialScaling(E_poly)
        nu = torch.tensor(0.31)
        mu = temperature.PolynomialScaling(E_poly / (2 * (1 + nu))).value(
            T=torch.tensor(T)
        )
        a = 0.358e-9
        b = a / 2 * np.sqrt(1**2 + 1**2 + 0**2)
        eps_0 = eps_dot_0
        g = k * T / (mu.item() * 1.0e9 * b**3) * np.log(eps_0 / erate)
        other_gs.append(g)
        other_sigmas.append(ys / mu.item())
    """
    ff = np.poly1d(np.polyfit(other_gs[:], other_sigmas[:], deg=1))
    t = np.linspace(0, np.amax(other_gs[:]), 500)
    plt.plot(other_gs[:], other_sigmas[:], 'ko', label='800H Alloy')
    plt.plot(t, ff(t), 'k')
    plt.ylim([1e-4, 1e-2])
    plt.xlabel('g')
    plt.ylabel('${\sigma}_{f}/{\mu}$')
    plt.legend()
    plt.grid(True)
    # plt.savefig("800H_Diagram.png", dpi = 300)
    plt.show()
    plt.close()  
    """
    # creep data
    fnames = glob.glob(path + os.path.join(full_creep_dir, "*.csv"))
    creep_temps = []
    erates = []
    strain_index = []
    yield_strength = []
    primary_rates = []
    for fn in fnames:
        with open(fn, "r") as f:
            metadata = f.readline().strip().split(",")
            creep_temps.append(float(metadata[4]))
            yield_strength.append(float(metadata[5]))
            data = np.loadtxt(f, delimiter=",")
            time = data[:, 0]
            strain = data[:, 1]
            rates = np.gradient(strain, time)
            real_index = np.where((rates > 0) & (rates != float("inf")))
            index = real_index[0][0]
            primary_rates.append(rates[index])

    creep_gs = []
    creep_sigmas = []
    for erate, ys, T in zip(primary_rates, yield_strength, creep_temps):
        k = 1.38064e-20
        E_poly = torch.tensor([-3.94833389e-02, -5.20197047e01, 1.95594836e05])
        E = temperature.PolynomialScaling(E_poly)
        nu = torch.tensor(0.31)
        mu = temperature.PolynomialScaling(E_poly / (2 * (1 + nu))).value(
            T=torch.tensor(T)
        )
        a = 0.358e-9
        b = a / 2 * np.sqrt(1**2 + 1**2 + 0**2)
        eps_0 = eps_dot_0
        g = k * T / (mu.item() * 1.0e9 * b**3) * np.log(eps_0 / float(erate))
        creep_gs.append(g)
        creep_sigmas.append(ys / mu.item())

    total_erates = np.concatenate((erates, other_erates, primary_rates))
    total_gs = np.concatenate((gs[:], other_gs[:], creep_gs))
    total_sigmas = np.concatenate((sigmas[:], other_sigmas[:], creep_sigmas))
    total_temps = np.concatenate((temps, other_temps, creep_temps))

    return total_erates, total_gs, total_sigmas, total_temps


if __name__ == "__main__":

    total_erates, total_gs, total_sigmas, total_temps = make_data()

    ff = np.poly1d(np.polyfit(total_gs[:], total_sigmas[:], deg=2))
    t = np.linspace(0, np.amax(total_gs[:]), 500)
    plt.semilogy(total_gs[:], total_sigmas[:], "ko", label="800H Alloy")
    plt.semilogy(t, ff(t), "k")
    # plt.xlim([0.0, 1.5])
    plt.ylim([1e-5, 1e-2])
    plt.xlabel("g")
    plt.ylabel("${\sigma}_{f}/{\mu}$")
    plt.legend()
    plt.grid(True)
    # plt.savefig("800H_Diagram_total_polyfit.png", dpi = 300)
    plt.show()
    plt.close()

    # define bilinear diagram model
    g_crit = 0.4

    def diagram(gs, A=-2.35, C=-2.4921, g0=g_crit):
        B = C - A * g0
        yvalues = []
        for g in gs:
            if g <= g0:
                yvalues.append(np.power(10, C))
            else:
                yvalues.append(np.power(10, A * g + B))
        return np.array(yvalues)

    plt.semilogy(total_gs[:], total_sigmas[:], "ko", label="800H Alloy")
    plt.semilogy(np.sort(total_gs), diagram(np.sort(total_gs)), "k")
    # plt.xlim([0.0, 1.5])
    plt.ylim([1e-5, 1e-2])
    plt.xlabel("g")
    plt.ylabel("${\sigma}_{f}/{\mu}$")
    plt.legend()
    plt.grid(True)
    # plt.savefig("800H_Diagram_total.png", dpi = 300)
    plt.show()
    plt.close()

    for i, erate in enumerate(total_erates):
        if erate < 1e-7:
            if total_gs[i] <= g_crit:
                plt.semilogy(total_gs[i], total_sigmas[i], "ro")
            else:
                plt.semilogy(total_gs[i], total_sigmas[i], "rx")
        elif (erate < 1e-6) and (erate >= 1e-7):
            if total_gs[i] <= g_crit:
                plt.semilogy(total_gs[i], total_sigmas[i], "bo")
            else:
                plt.semilogy(total_gs[i], total_sigmas[i], "bx")
        elif (erate < 1e-5) and (erate >= 1e-6):
            if total_gs[i] <= g_crit:
                plt.semilogy(total_gs[i], total_sigmas[i], "go")
            else:
                plt.semilogy(total_gs[i], total_sigmas[i], "gx")
        elif (erate < 1e-4) and (erate >= 1e-5):
            if total_gs[i] <= g_crit:
                plt.semilogy(total_gs[i], total_sigmas[i], "co")
            else:
                plt.semilogy(total_gs[i], total_sigmas[i], "cx")
        elif (erate < 1e-3) and (erate >= 1e-4):
            if total_gs[i] <= g_crit:
                plt.semilogy(total_gs[i], total_sigmas[i], "mo")
            else:
                plt.semilogy(total_gs[i], total_sigmas[i], "mx")
        elif (erate < 1e-2) and (erate >= 1e-3):
            if total_gs[i] <= g_crit:
                plt.semilogy(total_gs[i], total_sigmas[i], "yo")
            else:
                plt.semilogy(total_gs[i], total_sigmas[i], "yx")
        elif erate >= 1e-2:
            if total_gs[i] <= g_crit:
                plt.semilogy(total_gs[i], total_sigmas[i], "ko")
            else:
                plt.semilogy(total_gs[i], total_sigmas[i], "kx")

    plt.semilogy(np.sort(total_gs), diagram(np.sort(total_gs)), "k", lw=3.0)
    plt.semilogy(
        np.array([g_crit] * len(total_gs)), diagram(np.sort(total_gs)), "k--", lw=3.0
    )
    # plt.xlim([0.0, 1.5])
    plt.ylim([1e-4, 1e-2])
    plt.xlabel("g")
    plt.ylabel("${\sigma}_{f}/{\mu}$")
    plt.legend(
        [
            Line2D([0], [0], color="r", marker="o"),
            Line2D([0], [0], color="b", marker="o"),
            Line2D([0], [0], color="g", marker="o"),
            Line2D([0], [0], color="c", marker="o"),
            Line2D([0], [0], color="m", marker="o"),
            Line2D([0], [0], color="y", marker="o"),
            Line2D([0], [0], color="k", marker="o"),
            Line2D([0], [0], color="k", lw=4),
            Patch(facecolor="k", edgecolor=None, alpha=0.5),
        ],
        [
            "${\dot{\epsilon}<1e-7}$",
            "${\dot{\epsilon}}=[1e-7, 1e-6]$",
            "${\dot{\epsilon}}=[1e-6, 1e-5]$",
            "${\dot{\epsilon}}=[1e-5, 1e-4]$",
            "${\dot{\epsilon}}=[1e-4, 1e-3]$",
            "${\dot{\epsilon}}=[1e-3, 1e-2]$",
            "${\dot{\epsilon}>=1e-2}$",
            "bilinear fitting",
        ],
        loc="best",
    )
    plt.grid(True)
    # plt.savefig("800H_Diagram_bilinear.png", dpi = 300)
    plt.show()
    plt.close()
