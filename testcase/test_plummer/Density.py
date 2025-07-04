import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting tools
import argparse    

parser = argparse.ArgumentParser()
parser.add_argument("-F")
args = parser.parse_args()
N = int(args.F)
filename = "DATA/%05d" % N
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

distances = np.linalg.norm(positions, axis=1)

counts, bin_edges = np.histogram(distances, bins=50, range=[0, 100])

r = 0.5*(bin_edges[:-1] + bin_edges[1:])
a = 20
M = 20
rho = counts * (M / np.shape(positions)[0]) / ((4 * np.pi / 3) * (bin_edges[1:]**3 - bin_edges[:-1]**3))
plt.figure(figsize=(8,5))
plt.plot(r, rho, marker='o', linestyle='-', label='Density')


plt.plot(r, 3 * M / (4 * np.pi * a**3) / (1 + r**2/a**2)**2.5, label='Theoretical')
plt.xlabel('Distance to Origin')
plt.ylabel('Density')
plt.title('Density Distribution of Particle Distances (Line Plot)')
plt.legend()


plt.savefig("density.png")
