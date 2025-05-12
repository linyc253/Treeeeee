import numpy as np
import random
import argparse

# Parameters
G = 1                       # Newton's gravity constant
M = 20                      # total mass
a = 250                     # disk scale radius
b = 35                      # disk scale height
c = 50                      # bulge scale radius

parser = argparse.ArgumentParser()
parser.add_argument("-N", "--N", help = "Number of particles")
args = parser.parse_args()
N = int(args.N)             # number of total particles

# Calculated quantities
m = M/N                     # mass of each particle
N_d = np.floor(0.8 * N)
N_b = N - N_d
M_d = M * N_d/N
M_b = M - M_d

# Calculate Miyamoto–Nagai disk with N_d particles
def rho_mn(r, z, M, a, b):
    """
    Miyamoto-Nagai disk *density* ρ(r,z).
    """
    d = np.sqrt(b*b + z*z)
    num = b**2 * M * (a*r**2 + (a + 3*d)*(a + d)**2)
    den = 4*np.pi * d**3 * (r**2 + (a + d)**2)**(2.5)
    return num/den

def sample_mn_disk_rejection(N, M, a, b, c=None, d=None,
                             rmax=5.0, zmax=5.0, nr=200, nz=200):
    """
    Sample N points in a Miyamoto–Nagai disk via rejection sampling.
    - M, a, b : MN parameters
    - c ~ a, d ~ b : envelope scales
    - rmax, zmax : grid bounds for estimating C
    - nr, nz       : grid resolution for C
    Returns: array shape (N,3) of (x,y,z)
    """
    # 1) set envelope scales if not given
    c = a if c is None else c
    d = b if d is None else d

    # 2) build grid and estimate envelope constant C
    rr = np.linspace(0, rmax, nr)
    zz = np.linspace(-zmax, zmax, nz)
    Rg, Zg = np.meshgrid(rr, zz, indexing='xy')
    
    # target*Jacobian = r * rho
    target = Rg * rho_mn(Rg, Zg, M, a, b)
    # envelope pdf = f_r(r) * f_z(z)
    f_r = c * Rg / (Rg*Rg + c*c)**1.5
    f_z = (d/np.pi) / (Zg*Zg + d*d)
    env_pdf = f_r * f_z

    C = np.max(target/env_pdf)

    # 3) rejection loop
    samples = []
    while len(samples) < N:
        # sample r from f_r: CDF F_r(r)=1 - c/sqrt(r^2+c^2) -> invert
        u = np.random.rand(N - len(samples))
        r_cand = c * np.sqrt(1./(1.-u)**2 - 1.)
        # sample z from f_z: CDF F_z(z)=0.5 + (1/pi) arctan(z/d) -> invert
        u2 = np.random.rand(N - len(samples))
        z_cand = d * np.tan(np.pi*(u2 - 0.5))
        # sample phi uniformly
        phi = np.random.uniform(0, 2*np.pi, size=r_cand.size)

        # compute acceptance ratio
        targ = r_cand * rho_mn(r_cand, z_cand, M, a, b)
        env  = (c * r_cand/(r_cand**2 + c*c)**1.5) * ((d/np.pi)/(z_cand**2 + d*d))
        acc_prob = targ / (C * env)

        # accept
        u3 = np.random.rand(r_cand.size)
        keep = u3 < acc_prob
        rs = r_cand[keep]
        phs= phi[keep]
        zs = z_cand[keep]

        xs = rs * np.cos(phs)
        ys = rs * np.sin(phs)
        for x,y,z in zip(xs, ys, zs):
            samples.append((x,y,z))
     
    return np.array(samples[:N])

disk = sample_mn_disk_rejection(N_d, M_d, a, b)

# Calculate Hernquist bulge with N_b particles
phi = np.random.rand(N_b)
theta = np.arccos( np.random.rand(-1,1, size = N_b) )
r = c / (np.random.rand(N_b) - 1)

xb = r*np.sin(theta)*np.cos(phi)
yb = r*np.sin(theta)*np.sin(phi)
zb = r*np.cos(theta)

bulge = []
for x,y,z in zip(xb,yb,zb):
    bulge.append( (x,y,z) )

# for i in range(N_b):
#     # establish random points following distribution function in spherical
#     phi[i] = np.random.uniform(0, 2*np.pi)
#     theta[i] = np.arccos( np.random.uniform(-1,1) )
#     r[i] = c / (np.random.uniform(0,1)**(-0.50)-1)
#     # transform to Cartesian coordinate
#     x[i] = r[i] * np.sin(theta[i]) * np.cos(phi[i])
#     y[i] = r[i] * np.sin(theta[i]) * np.sin(phi[i])
#     z[i] = r[i] * np.cos(theta[i])

# Calculate velocity by rejection method
def sample_velocity(v_esc):
    while True:
        q = np.random.uniform(0, 1)  # trial q in [0,1]
        gq = q**2 * (1 - q**2)**(7/2)
        if np.random.uniform(0, 1) < gq / 0.1:  # 0.1 is approx max of g(q)
            return q * v_esc


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
print(disk)
print(bulge)

for i in range(N):
    print('%15f %16f %17f' % (vx[i], vy[i], vz[i]))
