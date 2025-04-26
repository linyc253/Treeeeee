import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting tools

# Load the file
with open('Final.dat', 'r') as f:
    lines = f.readlines()

# Parse the number of particles
num_particles = int(lines[0].strip())

# Masses: lines 2 to 4
# masses = np.array([float(lines[i].strip()) for i in range(1, 1 + num_particles)])

# Positions: lines 5 to 7
positions = np.array([
    list(map(float, lines[i].strip().split()))
    for i in range(1 + num_particles, 1 + 2 * num_particles)
])

# Velocities: lines 8 to 10
# velocities = np.array([
#     list(map(float, lines[i].strip().split()))
#     for i in range(1 + 2 * num_particles, 1 + 3 * num_particles)
# ])

# Create a new figure
fig = plt.figure(figsize=(8, 6))
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

# Optional: set equal aspect ratio (good if your data is physical space)
ax.set_box_aspect([np.ptp(positions[:,0]), np.ptp(positions[:,1]), np.ptp(positions[:,2])])

# Show the plot
plt.tight_layout()
plt.show()
