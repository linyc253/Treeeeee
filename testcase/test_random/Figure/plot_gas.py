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
    num_particles = int(lines[0].strip())

    # Masses:
    masses = np.array([float(lines[i].strip()) for i in range(1, 1 + num_particles)])

    # Positions:
    positions = np.array([
        list(map(float, lines[i].strip().split()))
        for i in range(1 + num_particles, 1 + 2 * num_particles)
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
        c=masses,
        cmap='YlOrRd',
        vmin=0.1, vmax=0.4,
        s=1,
        alpha=0.7
    )

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Fix the axes
    ax.axes.set_xlim3d(left=-300, right=300)
    ax.axes.set_ylim3d(bottom=-300, top=300)
    ax.axes.set_zlim3d(bottom=-300, top=300)
    ax.set_box_aspect([1,1,1])
    ax.set_axis_off()
    

    # Save figure
    figurename = "%05d" % i
    fig.savefig( figurename+".png")
    plt.close()
