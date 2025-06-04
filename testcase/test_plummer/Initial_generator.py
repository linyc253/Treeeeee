import numpy as np
import random
import argparse

# Parameters
G = 1                       # Newton's gravity constant
a = 20                      # critical radius
M = 20                      # total mass

parser = argparse.ArgumentParser()
parser.add_argument("-N", "--N", help = "Number of particles")
args = parser.parse_args()
N = int(args.N)             # number of total particles

# Calculated quantities
m = M/N                     # mass of each particle

# Calculate velocity by rejection method
def sample_velocity(v_esc):
    while True:
        q = np.random.uniform(0, 1)  # trial q in [0,1]
        gq = q**2 * (1 - q**2)**(7/2)
        if np.random.uniform(0, 1) < gq / 0.1:  # 0.1 is approx max of g(q)
            return q * v_esc

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
vec = np.empty(N)
vel = np.empty(N)       # set up spherical coordinate in velocity space
v_theta = np.empty(N)
v_phi = np.empty(N)
vx = np.empty(N)        # Cartesian coordinate in velocity space
vy = np.empty(N)
vz = np.empty(N)

for i in range(N):
    # calculate the speed by energy conservation
    vec[i] = (2*G*M)**0.5 * (r[i]**2 + a**2)**(-0.25)
    vel[i] = sample_velocity(vec[i])
    # attribute the velocity direction randomly
    v_phi[i] = np.random.uniform(0, 2*np.pi)
    v_theta[i] = np.arccos( np.random.uniform(-1,1) )
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
