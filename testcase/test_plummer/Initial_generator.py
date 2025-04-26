import numpy as np
import random

# Generation of the N particle are decoupled, which can be parallel.
# When parallel, position need to be done (at least r[i]) before calculating velocity.

# Parameters
G = 1           # Newton's gravity constant
N = 10000       # number of total particles
a = 0.375       # critical radius
M = 1           # total mass

# Calculated quantities
m = M/N         # mass of each particle

# Generate position of N random particles following distribution
r = np.empty(N)         # set up the spherical coordinate
theta = np.empty(N)
phi = np.empty(N)
x = np.empty(N)         # set up the Cartesian coordinate 
y = np.empty(N)
z = np.empty(N)

for i in range(N):
    # establish random points following distribution function in spherical
    phi[i] = np.random.uniform(0, 2*np.pi)
    theta[i] = np.arccos( np.random.uniform(-1,1) )
    r[i] = a / np.sqrt( np.random.uniform(0, 1)**(-2.0 / 3.0) - 1)
    # transform to Cartesian coordinate
    x[i] = r[i] * np.sin(theta[i]) * np.cos(phi[i])
    y[i] = r[i] * np.sin(theta[i]) * np.sin(phi[i])
    z[i] = r[i] * np.cos(theta[i])

# Generate the velocity of the N particles by energy conservation
vel = np.empty(N)       # set up spherical coordinate in velocity space
v_theta = np.empty(N)
v_phi = np.empty(N)
vx = np.empty(N)        # Cartesian coordinate in velocity space
vy = np.empty(N)
vz = np.empty(N)

for i in range(N):
    # calculate the speed by energy conservation
    vel[i] = (2*G*M/m) * (r[i]**2 + a**2)**(-0.25)
    # attribute the velocity direction randomly
    v_phi[i] = np.random.uniform(0, 2*np.pi)
    v_theta[i] = theta = np.arccos( np.random.uniform(-1,1) )
    # turn to Cartesian coordinate in velocity space
    vx[i] = vel[i] * np.sin(v_theta[i]) * np.cos(v_phi[i])
    vy[i] = vel[i] * np.sin(v_theta[i]) * np.sin(v_phi[i])
    vz[i] = vel[i] * np.cos(v_theta[i])

# Print the result formally
print(N)
for i in range(N):
    print(m)
for i in range(N):
    print('%12f %13f %14f' % (x[i], y[i], z[i]))
for i in range(N):
    print('%15f %16f %17f' % (vx[i], vy[i], vz[i]))