import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from matplotlib.ticker import NullFormatter
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams['font.family'] = "STIXGeneral"
plt.rcParams['font.size'] = 14

Data = np.loadtxt("comparison.dat", skiprows=1)
plt.plot(Data[:, 0], Data[:, 1], '.-', c='k', label="Brute Force")
plt.plot(Data[:, 0], Data[:, 2], '.-', c='r', label="Tree Algorithm")
plt.title("Time Elapsed of Tree Algorithm Compared with Brute Force")
plt.xlabel("$N$")
plt.ylabel("Time Elapsed (ms)")
plt.legend()
plt.savefig("time.png", transparent=True)