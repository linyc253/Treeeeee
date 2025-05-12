import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os

# Parameter
data_dir = "../"
# parser = argparse.ArgumentParser()
# parser.add_argument("-N", help = "Number of figures")
# args = parser.parse_args()
# N_f = int(args.N)       

# Load the file
# for i in range(1, N_f):
#     filename = "{:05d}.dat"
#     # Load the file
#     filepath = os.path.join(data_dir, filename.format(i))
#     with open(filepath, 'r') as f:
#         lines = f.readlines()

#     # Parse the number of particles
#     num_particles = int(lines[0].strip())

#     # Positions:
#     positions = np.array([
#         list(map(float, lines[i].strip().split()))
#         for i in range(1 + num_particles, 1 + 2 * num_particles)
#     ])

#     # Create a new figure
#     fig = plt.figure(figsize=(8, 8))
#     ax = fig.add_subplot(111, projection='3d')

#     # Scatter plot for the positions
#     ax.scatter(
#         positions[:, 0],  # x-coordinates
#         positions[:, 1],  # y-coordinates
#         positions[:, 2],  # z-coordinates
#         s=1,              # marker size (small for many particles)
#         alpha=0.7         # transparency for better visibility
#     )

#     # Set labels
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')

#     # Fix the axes
#     ax.axes.set_xlim3d(left=-100, right=100)
#     ax.axes.set_ylim3d(bottom=-100, top=100)
#     ax.axes.set_zlim3d(bottom=-100, top=100)
#     ax.set_box_aspect([1,1,1])

#     # Save figure
#     figurename = "%05d" % i
#     fig.savefig( figurename+".png")
#     plt.close()

filename = "Initial.dat"
# Load the file
filepath = os.path.join(data_dir, filename)
with open(filepath, 'r') as f:
    lines = f.readlines()

# Parse the number of particles
num_particles = int(lines[0].strip())

# Positions:
positions = np.array([
    list(map(float, lines[i].strip().split()))
    for i in range(1 + num_particles, 1 + 2 * num_particles)
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
ax.axes.set_xlim3d(left=-1000, right=1000)
ax.axes.set_ylim3d(bottom=-1000, top=1000)
ax.axes.set_zlim3d(bottom=-1000, top=1000)
ax.set_box_aspect([1,1,1])

plt.show()