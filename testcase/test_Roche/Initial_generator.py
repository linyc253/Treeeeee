import numpy as np
import random
import argparse

# Parameters
G = 1                       # Newton's gravity constant
b = 500                      # collision parameter
M = 10000                   # total mass of the primary
k = 1                       # initial distance

# Loading input parameter
parser = argparse.ArgumentParser()
parser.add_argument("-N", "--N", help = "Number of particles")
args = parser.parse_args()
N = int(args.N)             # number of particles for the primary

# Calculated quantities
m1 = M/N                    # mass of each particle for primary
m2 = m1 * (5e-5)            # mass of each particle for secondary
n = int(N/50)              # the npt of the secondary star is 1/100 of the primary
L = 0.1*b                   # the closest distance between the stars
c = L/100                   # critical radius for the secondary
a = 24*c                    # critical radius for the primary
v = np.sqrt(G*M*L*(np.sqrt(k**2+1)*b-L)/(np.sqrt(k**2+1)*b*(b**2-L**2))) # initial velocity

# Generate position of the two plummers
def sample_position(N,a):
    phi = np.empty(N)
    theta = np.empty(N)
    r = np.empty(N)
    x = np.empty(N)
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
    return x,y,z

x1, y1, z1 = sample_position(N,a)
x2, y2, z2 = sample_position(n,c)
x2 -= k*b
y2 -= b

# Generate the velocity of the two plummers

# 1) Calculate velocity by rejection method
def sample_velocity(vec):
    while True:
        q = np.random.uniform(0, 1)  # trial q in [0,1]
        gq = q**2 * (1 - q**2)**(7/2)
        if np.random.uniform(0, 1) < gq / 0.1:  # 0.1 is approx max of g(q)
            return q * vec

# 2) Calculate the escaping velocity
def get_velocity(x,y,z,n,m):
    r = np.empty(n)
    vec = np.empty(n)
    vel = np.empty(n)
    v_phi = np.empty(n)
    v_theta = np.empty(n)
    vx = np.empty(n)
    vy = np.empty(n)
    vz = np.empty(n)
    for i in range(n):
        r[i] = np.sqrt(x[i]**2 + y[i]**2 + z[i]**2)
        # calculate the speed by energy conservation
        vec[i] = (2*G*m)**0.5 * (r[i]**2 + a**2)**(-0.25)
        vel[i] = sample_velocity(vec[i])
        # attribute the velocity direction randomly
        v_phi[i] = np.random.uniform(0, 2*np.pi)
        v_theta[i] = np.arccos( np.random.uniform(-1,1) )
        # turn to Cartesian coordinate in velocity space
        vx[i] = vel[i] * np.sin(v_theta[i]) * np.cos(v_phi[i])
        vy[i] = vel[i] * np.sin(v_theta[i]) * np.sin(v_phi[i])
        vz[i] = vel[i] * np.cos(v_theta[i])
    return vx, vy, vz

vx1, vy1, vz1 = get_velocity(x1,y1,z1,N,M)
vx2, vy2, vz2 = get_velocity(x2,y2,z2,n,M*1e-6)
vx2 += v

# Print the result formally
print(N+n)
for i in range(N):
    print(m1)
for i in range(n):
    print(m2)
for i in range(N):
    print('%12f %13f %14f' % (x1[i], y1[i], z1[i]))
for i in range(n):
    print('%12f %13f %14f' % (x2[i], y2[i], z2[i]))
for i in range(N):
    print('%15f %16f %17f' % (vx1[i], vy1[i], vz1[i]))
for i in range(n):
    print('%15f %16f %17f' % (vx2[i], vy2[i], vz2[i]))