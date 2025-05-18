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
args = parser.parse_args()
halo = int(args.H)
N_f = int(args.F)/2
N_f = int(N_f)

# Load the file
for i in range(1,1+N_f):
    filename = "{:05d}.dat"
    # Load the file
    filepath = os.path.join(data_dir, filename.format(i))
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Parse the number of particles
    num_particles = int(lines[0].strip())
    if halo == 1:
        num_particles = int(num_particles/5)

    # Positions:
    if halo == 1:
        positions = np.array([
            list(map(float, lines[i].strip().split()))
            for i in range(1 + 5*num_particles, 1 + 6 * num_particles)
        ])
    else:
        positions = np.array([
            list(map(float, lines[i].strip().split()))
            for i in range(1 + num_particles, 1 + 2 * num_particles)
        ])

    R = np.sqrt(positions[:,0]**2 + positions[:,1]**2)
    phi = np.arctan2(positions[:,1], positions[:,0])

    R_max = R.max()
    n_bins = 30
    R_edges = np.linspace(0, R_max, n_bins+1)
    R_centers = 0.5*(R_edges[:-1] + R_edges[1:])
    A2  = np.zeros(n_bins)

    for j in range(n_bins):
        sel = (R >= R_edges[j]) & (R < R_edges[j+1])
        Nj  = sel.sum()
        if Nj > 0:
            A2[j] = np.abs(np.sum(np.exp(2j * phi[sel])))/Nj
        else:
            A2[j] = 0

    