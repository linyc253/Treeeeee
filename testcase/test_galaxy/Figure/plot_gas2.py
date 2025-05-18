import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os

# Parameter
data_dir = "../DATA/"
parser = argparse.ArgumentParser()
parser.add_argument("-N", help = "Number of figures")
args = parser.parse_args()
N_f = int(args.N)/2
N_f = int(N_f)       

# Load the file
for i in range(1 + N_f,1 + 2*N_f):
    filename = "{:05d}.dat"
    # Load the file
    filepath = os.path.join(data_dir, filename.format(i))
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Parse the number of particles
    num_particles = int(lines[0].strip())

    # Positions:
    positions = np.array([
        list(map(float, lines[i].strip().split()))
        for i in range(1 + num_particles, 1 + 2 * num_particles)
    ])

    # Force:
    force = np.array([
        list(map(float, lines[i].strip().split()))
        for i in range(1 + 3 * num_particles, 1 + 4 * num_particles)
    ])
    F = np.sqrt(force[:,0]**2+force[:,1]**2+force[:,2]**2)
    # np.clip(F,1.0e-4,np.max(F))
    F += 1.0e-14
    F = -np.log(F)

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
        c=F,
        cmap='YlOrRd',
        vmin = np.min(F), vmax=np.max(F)
    )

    # Fix the axes
    ax.axes.set_xlim3d(left=-2500, right=2500)
    ax.axes.set_ylim3d(bottom=-2500, top=2500)
    ax.axes.set_zlim3d(bottom=-2500, top=2500)
    ax.set_box_aspect([1,1,1])
    ax.set_axis_off()

    # Save figure
    figurename = "%05d" % i
    fig.savefig( figurename+".png")
    plt.close()