import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("Energy.dat", skiprows=1)

plt.figure()
plt.plot(data[:, 0])
plt.title("Kinetic Energy")
plt.savefig("Kinetic.png")
plt.figure()
plt.plot(data[:, 1])
plt.title("Potential Energy")
plt.savefig("Potential.png")
plt.figure()
plt.plot(data[:, 2])
plt.title("Total Energy")
plt.savefig("Energy.png")