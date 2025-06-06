import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import numpy as np
from matplotlib.ticker import NullFormatter, LogLocator
from scipy.optimize import curve_fit
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams['font.family'] = "STIXGeneral"
plt.rcParams['font.size'] = 14

def f(x, a, b):
    return a*x + b

if __name__ == "__main__":
    fig = plt.figure(figsize=(5, 5), dpi=200, constrained_layout=True)
    gs = fig.add_gridspec(1, 1)
    ax1 = fig.add_subplot(gs[0, 0])

    Data = np.loadtxt("comparison.dat", skiprows=1)
    ax1.scatter(Data[:, 0], Data[:, 1], s=20, c='w', label="dipole", zorder=10, edgecolors="k")
    ax1.scatter(Data[:, 0], Data[:, 2], s=20, c='w', label="quadrupole", zorder=10, edgecolors="r")

    popt, pconv = curve_fit(f, np.log10(Data[:, 0]), np.log10(Data[:, 1]))
    ax1.plot(10**np.linspace(-3, 1, 10), 10**f(np.linspace(-3, 1, 10), popt[0], popt[1]), c='k', lw=1)
    print(popt)

    popt, pconv = curve_fit(f, np.log10(Data[:, 0]), np.log10(Data[:, 2]))
    ax1.plot(10**np.linspace(-3, 1, 10), 10**f(np.linspace(-3, 1, 10), popt[0], popt[1]), c='r', lw=1)
    print(popt)

    ax1.set_yscale("log")
    ax1.set_xscale("log")
    ax1.set_title("Error compared to Brute Force")
    ax1.set_xlabel(r"$\theta$", fontsize=16)
    ax1.set_ylabel("error", fontsize=16)
    ax1.legend(fontsize=14)
    ax1.set_xlim(0.05, 2)
    ax1.set_ylim(10**-8, 1)
    ax1.set_xticks([0.1, 1])
    ax1.set_xticklabels([0.1, 1], fontsize=14)
    locmin = LogLocator(base=100.0, subs=np.linspace(1, 9, 9))
    ax1.yaxis.set_minor_locator(locmin)
    ax1.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    plt.savefig("error.pdf" , transparent=True)