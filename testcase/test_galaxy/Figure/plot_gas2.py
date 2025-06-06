import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os

# Parameter
data_dir = "../DATA/"
parser = argparse.ArgumentParser()
parser.add_argument("-F", help = "Number of figures")
parser.add_argument("-H", default=0, help = "Turn on/off the halo. Default:0.")
parser.add_argument("-B", default=0, help = "Existence of SMBN. Default:0.")
args = parser.parse_args()
halo = int(args.H)
BH = int(args.B)
N_f = int(args.F)/2
N_f = int(N_f)

# Load the file
for i in range(1+N_f,1+2*N_f):
    filename = "{:05d}.dat"
    # Load the file
    filepath = os.path.join(data_dir, filename.format(i))
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Parse the number of particles
    num_particles = int(lines[0].strip())
    if halo == 1:
        if BH == 1:
            num_particles -= 1
        num_particles = int(num_particles/5)

    # Positions:
    if halo == 1:
        positions = np.array([
            list(map(float, lines[i].strip().split()))
            if BH == 1:
                for i in range(2 + 5*num_particles, 2 + 6 * num_particles)
            else:
                for i in range(1 + 5*num_particles, 1 + 6 * num_particles)
        ])
    else:
        positions = np.array([
            list(map(float, lines[i].strip().split()))
            for i in range(1 + num_particles, 1 + 2 * num_particles)
        ])

    # # Force:
    # force = np.array([
    #     list(map(float, lines[i].strip().split()))
    #     for i in range(1 + 3 * num_particles, 1 + 4 * num_particles)
    # ])
    # F = np.sqrt(force[:,0]**2+force[:,1]**2+force[:,2]**2)
    # # np.clip(F,1.0e-4,np.max(F))
    # F += 1.0e-14
    # F = -np.log(F)

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
        c=np.zeros(num_particles),
        cmap='YlOrRd',
        vmin = -0.2, vmax=1.0
    )

    # Fix the axes
    ax.axes.set_xlim3d(left=-800, right=800)
    ax.axes.set_ylim3d(bottom=-800, top=800)
    ax.axes.set_zlim3d(bottom=-800, top=800)
    ax.set_box_aspect([1,1,1])
    ax.set_axis_off()

    # Save figure
    figurename = "%05d" % i
    fig.savefig( figurename+".png")
    plt.close()
