import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting tools
import argparse

# Read parameter from script
# parser = argparse.ArgumentParser()
# parser.add_argument("-N", "-N_f", help = "Number of figures")
# args = parser.parse_args()
# N_f = int(args.N_f)

N_f = 126

parser = argparse.ArgumentParser()
parser.add_argument("-D", "--N", help = "Distance")
args = parser.parse_args()
D = float(args.N)

num_particles = 2

# Load the file
for i in range(N_f):
    filename = "%05d" % i
    # Load the file
    with open(filename+".dat", 'r') as f:
        lines = f.readlines()

    # Positions:
    positions = np.array([
        list(map(float, lines[i].strip().split()))
        for i in range(3, 5)
    ])

    # Create a new figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot for the positions
    ax.scatter(
        positions[0, 0],  # x-coordinates
        positions[0, 1],  # y-coordinates
        positions[0, 2],  # z-coordinates
        s=50,             # marker size (small for many particles)
        alpha=0.7         # transparency for better visibility
    )
    ax.scatter(positions[1,0], positions[1,1], positions[1,2], s=5, alpha=0.7)

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Fix the axes
    ax.axes.set_xlim3d(left=-D*1.1, right=D*1.1)
    ax.axes.set_ylim3d(bottom=-D*1.1, top=D*1.1)
    ax.axes.set_zlim3d(bottom=-D*1.1, top=D*1.1)
    
    ax.set_box_aspect([1,1,1])
    fig.savefig(filename + ".png")
    plt.close()