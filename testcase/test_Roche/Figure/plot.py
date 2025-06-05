import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting tools
import argparse
import os

# Parameter
data_dir = "../DATA/"
parser = argparse.ArgumentParser()
parser.add_argument("-F")
args = parser.parse_args()
N_f = int(args.F)

# Load the file
for i in range(1, 1+N_f):
    filename = "{:05d}.dat"
    # Load the file
    filepath = os.path.join(data_dir, filename.format(i))
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Parse the number of particles
    Npt = int(lines[0].strip())

    # Positions:
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
        positions[0:int(Npt/1.02), 0],  # x-coordinates
        positions[0:int(Npt/1.02), 1],  # y-coordinates
        positions[0:int(Npt/1.02), 2],  # z-coordinates
        s=0.1,            # marker size (small for many particles)
        alpha=0.7,        # transparency for better visibility
        c=np.zeros(int(Npt/1.02)),
        cmap='YlOrRd',
        vmin = -0.2, vmax=1.0
    )
    ax.scatter(
        positions[int(Npt/1.02):Npt, 0],  # x-coordinates
        positions[int(Npt/1.02):Npt, 1],  # y-coordinates
        positions[int(Npt/1.02):Npt, 2],  # z-coordinates
        s=0.1,            # marker size (small for many particles)
        alpha=0.7,        # transparency for better visibility
        c=np.zeros(int(Npt*0.02/1.02)),
        cmap='YlOrRd',
        vmin = -1.0, vmax=0.2
    )

    # Fix the axes
    ax.axes.set_xlim3d(left=-400, right=400)
    ax.axes.set_ylim3d(bottom=-400, top=400)
    ax.axes.set_zlim3d(bottom=-400, top=400)
    ax.set_box_aspect([1,1,1])
    ax.set_axis_off()

    # Save figure
    figurename = "%05d" % i
    fig.savefig( figurename+".png")
    plt.close()
