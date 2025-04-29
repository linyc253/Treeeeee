import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from matplotlib.ticker import NullFormatter
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams['font.family'] = "STIXGeneral"
plt.rcParams['font.size'] = 14

Data = np.loadtxt("comparison.dat", skiprows=1)
plt.plot(Data[:, 0], Data[:, 1], '.-', c='k', label="dipole")
plt.plot(Data[:, 0], Data[:, 2], '.-', c='r', label="quadrupole")
plt.title("Error of Tree Algorithm compared to Brute Force")
plt.xlabel("$\\theta$")
plt.ylabel("error")
plt.legend()
plt.savefig("error.png")