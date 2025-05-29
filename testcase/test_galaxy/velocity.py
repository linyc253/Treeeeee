import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# Parameter
data_dir = "../DATA/"
parser = argparse.ArgumentParser()
parser.add_argument("-H", default=0, help = "Turn on/off the halo. Default:0.")
args = parser.parse_args()
halo = int(args.H)

# Load the file
filename = "Initial.dat"
with open(filename, 'r') as f:
    lines = f.readlines()

# Parse the number of particles
num_particles = int(lines[0].strip())
if halo == 1:
    num_particles = int(num_particles/5)

# Positions:
if halo == 1:
    r = np.array([
        list(map(float, lines[i].strip().split()))
        for i in range(1 + 5*num_particles, 1 + 6 * num_particles)
    ])
else:
    r = np.array([
        list(map(float, lines[i].strip().split()))
        for i in range(1 + num_particles, 1 + 2 * num_particles)
    ])
r_abs = np.zeros(num_particles)
for i in range(3):
    r_abs += r[:,i]**2
r_abs = np.sqrt(r_abs)

# Velocity:
if halo == 1:
    v = np.array([
        list(map(float, lines[i].strip().split()))
        for i in range(1 + 10 * num_particles, 1 + 11 * num_particles)
    ])
else:
    v = np.array([
        list(map(float, lines[i].strip().split()))
        for i in range(1 + 2 * num_particles, 1 + 3 * num_particles)
    ])

# Calculate tangent velocity
# v_tan = v-(v*x)x/(x*x)
v_t = np.empty((num_particles,3))
v_tan = np.zeros(num_particles)
for i in range(num_particles):
    for j in range(3):
        v_t[i,j] = v[i,j] - (v[i,:]*r[i,:]).sum()/(r[i,:]*r[i,:]).sum()*r[i,j]
        v_tan[i] += v_t[i,j]**2
v_tan = np.sqrt(v_tan)

# Create a new figure
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot()
curve = plt.scatter(r_abs,v_tan,s=0.1)
ax.set_xlim(0,7000)
ax.set_ylim(0,2.0)

# Save figure
fig.savefig( "velocity_curve.png" )
plt.close()
