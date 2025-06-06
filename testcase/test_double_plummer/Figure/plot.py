import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting tools
import argparse
import os

# Parameter
data_dir1 = "../DATA1/"
data_dir2 = "../DATA2/"
parser = argparse.ArgumentParser()
parser.add_argument("-F")
args = parser.parse_args()
N_f = int(args.F)

# Load the file
for i in range(1, 1+N_f):
    filename = "{:05d}.dat"
    # Load the file
    filepath1 = os.path.join(data_dir1, filename.format(i))
    with open(filepath1, 'r') as f:
        lines = f.readlines()

    # Parse the number of particles
    Npt = int(lines[0].strip())

    # Positions:
    positions1 = np.array([
        list(map(float, lines[i].strip().split()))
        for i in range(1 + Npt, 1 + 2 * Npt)
    ])

    # Load the file
    filepath2 = os.path.join(data_dir2, filename.format(i))
    with open(filepath2, 'r') as f:
        lines = f.readlines()

    # Parse the number of particles
    Npt = int(lines[0].strip())

    # Positions:
    positions2 = np.array([
        list(map(float, lines[i].strip().split()))
        for i in range(1 + Npt, 1 + 2 * Npt)
    ])

    # Create a new figure
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(1,2,1, projection='3d')
    ax2 = fig.add_subplot(1,2,2, projection='3d')
    plt.style.use('dark_background')

    # Make the panes and figure patch black
    fig.patch.set_facecolor('black')
    ax1.set_facecolor('black')
    ax2.set_facecolor('black')
    for axis in (ax1.xaxis, ax1.yaxis, ax1.zaxis):
        axis.pane.set_visible(False)
    ax1.grid(False)
    for axis in (ax2.xaxis, ax2.yaxis, ax2.zaxis):
        axis.pane.set_visible(False)
    ax2.grid(False)

    # Scatter plot for the positions
    ax1.scatter(
        positions1[:, 0],  # x-coordinates
        positions1[:, 1],  # y-coordinates
        positions1[:, 2],  # z-coordinates
        s=0.1,            # marker size (small for many particles)
        alpha=0.7,        # transparency for better visibility
        c=np.zeros(Npt),
        cmap='YlOrRd',
        vmin = -0.2, vmax=1.0
    )
    ax2.scatter(
        positions2[:, 0],  # x-coordinates
        positions2[:, 1],  # y-coordinates
        positions2[:, 2],  # z-coordinates
        s=0.1,            # marker size (small for many particles)
        alpha=0.7,        # transparency for better visibility
        c=np.zeros(Npt),
        cmap='YlOrRd',
        vmin = -0.2, vmax=1.0
    )

    # Fix the axes
    ax1.axes.set_xlim3d(left=-250, right=250)
    ax1.axes.set_ylim3d(bottom=-100, top=100)
    ax1.axes.set_zlim3d(bottom=-100, top=100)
    ax1.set_box_aspect([5,2,2])
    ax1.set_axis_off()
    ax1.set_title("Non-rotation", color='white')
    
    ax2.axes.set_xlim3d(left=-250, right=250)
    ax2.axes.set_ylim3d(bottom=-100, top=100)
    ax2.axes.set_zlim3d(bottom=-100, top=100)
    ax2.set_box_aspect([5,2,2])
    ax2.set_axis_off()
    ax2.set_title("Rotation", color='white')

    # Save figure
    figurename = "%05d" % i
    fig.savefig( figurename+".png")
    plt.close()
