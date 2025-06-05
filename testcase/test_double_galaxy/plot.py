import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

# Parameter
parser = argparse.ArgumentParser()
parser.add_argument("-H", default=0, help = "Turn on/off the halo. Default:0.")
parser.add_argument("-R", default=0.8, help = "Bulge, disk mass distribution ratio. Default:0.8")
args = parser.parse_args()
halo = int(args.H)
bd_rate = float(args.R)
k   = int(1+5*bd_rate)

# Load the file
filename = "Initial.dat"
# Load the file
with open(filename, 'r') as f:
    lines = f.readlines()

# Parse the number of particles
Npt = int(lines[0].strip())
if halo == 1:
    Npt = int(Npt/k)

# Positions:
if halo == 1:
    positions = np.array([
        list(map(float, lines[i].strip().split()))
        for i in range(1 + k*Npt, 1 + (k+1) * Npt)
    ])
else:
    positions = np.array([
        list(map(float, lines[i].strip().split()))
        for i in range(1 + Npt, 1 + 2 * Npt)
    ])

# Create a new figure
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
plt.style.use('dark_background')

# Make the panes and figure patch black
fig.patch.set_facecolor('black')
ax.set_facecolor('black')
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.pane.set_visible(False)
ax.grid(False)

# Scatter plot for the positions
ax.scatter(
    positions[:, 0],  # x-coordinates
    positions[:, 1],  # y-coordinates
    positions[:, 2],  # z-coordinates
    s=0.1,            # marker size (small for many particles)
    alpha=0.7,        # transparency for better visibility
    c=np.zeros(Npt),
    cmap='YlOrRd',
    vmin = -0.2, vmax=1.0
)

# Fix the axes
ax.axes.set_xlim3d(left=-2400, right=2400)
ax.axes.set_ylim3d(bottom=-800, top=800)
ax.axes.set_zlim3d(bottom=-800, top=800)
ax.set_box_aspect([3,1,1])
ax.set_axis_off()

# Save figure
fig.savefig("Initial.png", dpi=200)
plt.close()
