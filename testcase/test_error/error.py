import numpy as np

# Load the file
with open('Final_std.dat', 'r') as f:
    lines = f.readlines()
num_particles = int(lines[0].strip())
force_std = np.array([
    list(map(float, lines[i].strip().split()))
    for i in range(1 + 3 * num_particles, 1 + 4 * num_particles)
])


# Load the file
with open('Final.dat', 'r') as f:
    lines = f.readlines()
assert num_particles == int(lines[0].strip()), "number of particle mismatch"
force = np.array([
    list(map(float, lines[i].strip().split()))
    for i in range(1 + 3 * num_particles, 1 + 4 * num_particles)
])

# Follow the definition of Eq.(27) in https://iopscience.iop.org/article/10.1086/381391/fulltext/56750.text.html
e = np.sqrt(np.mean(np.linalg.norm(force - force_std, axis=1)**2 / np.linalg.norm(force, axis=1)**2))
print(e)