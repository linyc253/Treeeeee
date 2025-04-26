import numpy as np

# Load the file
with open('Initial.dat', 'r') as f:
    lines = f.readlines()

# Parse the number of particles
num_particles = int(lines[0].strip())

# Masses: lines 2 to 4
masses = np.array([float(lines[i].strip()) for i in range(1, 1 + num_particles)])

# Positions: lines 5 to 7
positions = np.array([
    list(map(float, lines[i].strip().split()))
    for i in range(1 + num_particles, 1 + 2 * num_particles)
])

# Velocities: lines 8 to 10
velocities = np.array([
    list(map(float, lines[i].strip().split()))
    for i in range(1 + 2 * num_particles, 1 + 3 * num_particles)
])

print("Masses array:")
print(masses)

print("\nPositions array:")
print(positions)

print("\nVelocities array:")
print(velocities)
