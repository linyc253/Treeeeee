import numpy as np
import random
import argparse

# Parameters
G = 1           # Newton's gravity constant
M = 30000       # mass of the sun
N = 9           # number of planets

# Generate masses of 9 planets
m = np.zeros(N+1)
m = [0.330, 4.87, 5.97, 0.642, 1898, 568, 86.8, 102, 0.0130, 1998000]
for i in range(N+1):
    m[i] *= 4*np.pi**2/1998000

# Generate position of 9 planets
x = np.zeros(N+1)
y = np.zeros(N+1)
z = np.zeros(N+1)
theta = np.zeros(N+1)

x = [57.9, 108.2, 149.6, 228.0, 778.5, 1432.0, 2867.0, 4515.0, 5906.4, 0]
for i in range(N):
    x[i] /= 149.6
theta = [7.0, 3.4, 0.0, 1.8, 1.3, 2.5, 0.8, 1.8, 17.2, 0]
for i in range(N):
    theta[i] *= np.pi/180
    z[i] = x[i] * np.sin(theta[i])
    x[i] *= np.cos(theta[i])


# Generate the velocity of N planets
vx = np.zeros(N+1)
vy = np.zeros(N+1)
vz = np.zeros(N+1)

vy = [47.4, 35.0, 29.8, 24.1, 13.1, 9.7, 6.8, 5.4, 4.7, 0]
for i in range(N):
    vy[i] *= 2*np.pi/29.8
    
# Print the result formally
print(N+1)
for i in range(N+1):
    print(m[i])
for i in range(N+1):
    print('%12f %13f %14f' % (x[i], y[i], z[i]))
for i in range(N+1):
    print('%15f %16f %17f' % (vx[i], vy[i], vz[i]))