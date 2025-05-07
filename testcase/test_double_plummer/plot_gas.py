import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting tools
import argparse

# Parameter
parser = argparse.ArgumentParser()
parser.add_argument("-N", "--N", help = "Number of figures")
args = parser.parse_args()
N_f = int(args.N)       

# Load the file
for i in range(1, N_f):
    filename = "%05d" % i
    # Load the file
    with open(filename+".dat", 'r') as f:
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

    # Velocities:
    velocities = np.array([
        list(map(float, lines[i].strip().split()))
        for i in range(1 + 2 * num_particles, 1 + 3 * num_particles)
    ])

    # Create a new figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot for the positions
    ax.scatter(
        positions[:, 0],  # x-coordinates
        positions[:, 1],  # y-coordinates
        positions[:, 2],  # z-coordinates
        s=1,              # marker size (small for many particles)
        alpha=0.7         # transparency for better visibility
    )

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Fix the axes
    ax.axes.set_xlim3d(left=-250, right=250)
    ax.axes.set_ylim3d(bottom=-100, top=100)
    ax.axes.set_zlim3d(bottom=-100, top=100)
    

    # Optional: set equal aspect ratio (good if your data is physical space)
    ax.set_box_aspect([5,2,2])
    fig.savefig(filename+".png")
    plt.close()
