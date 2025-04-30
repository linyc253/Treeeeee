import numpy as np
import random
import argparse

# Parameters
G = 1           # Newton's gravity constant
M = 1000        # mass of the sun
m = 100         # mass of the planet

# Read parameter from script
parser = argparse.ArgumentParser()
parser.add_argument("-D", "--N", help = "Distance between stars")
args = parser.parse_args()
D = float(args.N)         # distance between the stars

# Generate masses
mass = np.zeros(2)
mass = [M, m]

# Generate position
x = np.zeros(2)
y = np.zeros(2)
z = np.zeros(2)
x = [0, D]

# Generate the velocity
vx = np.zeros(2)
vy = np.zeros(2)
vz = np.zeros(2)
vy[0] = 0
vy[1] = (G*M**2/(M+m)/x[1])**0.5
    
# Print the result formally
print(2)
for i in range(2):
    print(mass[i])
for i in range(2):
    print('%12f %13f %14f' % (x[i], y[i], z[i]))
for i in range(2):
    print('%15f %16f %17f' % (vx[i], vy[i], vz[i]))