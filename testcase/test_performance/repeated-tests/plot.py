import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from matplotlib.ticker import NullFormatter
import argparse


plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams['font.family'] = "STIXGeneral"
plt.rcParams['font.size'] = 14


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-S", "--suffix", help = "suffix added to name")
    suffix = parser.parse_args().suffix

    fig = plt.figure(figsize=(5, 3), dpi=300, constrained_layout=True)
    gs = fig.add_gridspec(1, 1)
    ax1 = fig.add_subplot(gs[0, 0])

    Data = np.loadtxt("comparison-%s.dat" % suffix, skiprows=1)
    ax1.boxplot(Data[:, 1:].T, tick_labels=np.vectorize(int)(Data[:, 0]), showfliers=False)
    ax1.set_title(suffix)
    ax1.set_xlabel("$N$")
    ax1.set_ylabel("Time Elapsed (ms)")
    # ax1.legend()
    fig.savefig("time-%s.png" % suffix, transparent=True)