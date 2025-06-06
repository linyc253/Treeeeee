import numpy as np
import matplotlib.pyplot as plt

data1 = np.loadtxt("Energy1.dat", skiprows=1)
data2 = np.loadtxt("Energy2.dat", skiprows=1)

fig = plt.figure(figsize=(10, 4))

# Left subplot: Non‐rotation
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(data1[:, 0], 'g', label='Kinetic')
ax1.plot(data1[:, 1], 'r', label='Potential')
ax1.plot(data1[:, 2], 'b', label='Total Energy')
ax1.set_title("Non‐rotation")
ax1.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))

# Right subplot: Rotation
ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(data2[:, 0], 'g', label='Kinetic')
ax2.plot(data2[:, 1], 'r', label='Potential')
ax2.plot(data2[:, 2], 'b', label='Total Energy')
ax2.set_title("Rotation")
ax2.legend(loc='lower right', bbox_to_anchor=(1.0, 0.5))

# Figure‐level title
fig.suptitle("Double Plummer Energy")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("Plummer2_Energy.png")
plt.close()
