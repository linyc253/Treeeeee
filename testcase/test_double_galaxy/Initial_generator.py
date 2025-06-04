import numpy as np
import argparse

# Parameters
G = 1                       # Newton's gravity constant
M = 1000                    # total mass
a = 250                     # disk scale radius
b = 35                      # disk scale height
c = 50                      # bulge scale radius
e = 4*c                     # halo scale radius
distance = 3000             # distance between the two galaxy
angle    = 0                # the incline angle of two galaxy (rad)



parser = argparse.ArgumentParser()
parser.add_argument("-N", help = "Number of particles")
parser.add_argument("-S", default=0, help = "Turn on/off the spiral. Default:0.")
parser.add_argument("-H", default=0, help = "Turn on/off the halo. Default:0.")
args = parser.parse_args()
N       = int(args.N)               # number of total particles
spiral  = int(args.S)               # The switch of spiral
halo    = int(args.H)               # The switch of halo
distance = int(distance/2)

# Calculated quantities
m = M/N                     # mass of each particle
N_d = int(np.floor(0.8 * N))
N_b = N - N_d
N_h = 5 * N_d
M_d = M * N_d/N
M_b = M - M_d
M_h = 5 * M_d

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
    target = rho_mn(Rg, Zg, M, a, b)
    # envelope pdf = f_r(r) * f_z(z)
    f_r = c / (Rg*Rg + c*c)**1.5
    f_z = (d/np.pi) / (Zg*Zg + d*d)
    env_pdf = f_r * f_z

    C = np.max(target/env_pdf)

    # 3) rejection loop
    samples = []
    while len(samples) < N:
        # sample r from f_r: CDF F_r(r)=1 - c/sqrt(r^2+c^2) -> invert
        u = np.random.rand(N - len(samples))
        r_cand = c * np.sqrt(1/(1-u)**2 - 1)
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
        phs= phi[keep] + spiral * 0.01 * np.cos(2*phi[keep])
        zs = z_cand[keep]

        xs = rs * np.cos(phs)
        ys = rs * np.sin(phs)
        for x,y,z in zip(xs, ys, zs):
            samples.append((x,y,z))
     
    return np.array(samples[:N])

disk = sample_mn_disk_rejection(N_d, M_d, a, b)

# Calculate Plummer bulge with N_b particles
phi = np.random.uniform(0,2*np.pi, size = N_b)
theta = np.arccos( np.random.uniform(-1,1, size = N_b) )
r = c * np.sqrt( np.random.rand(N_b)**(-2.0/3.0)-1 )

xb = r*np.sin(theta)*np.cos(phi)
yb = r*np.sin(theta)*np.sin(phi)
zb = r*np.cos(theta)

bulge = []
for x,y,z in zip(xb,yb,zb):
    bulge.append( (x,y,z) )

# Calculate Plummer halo with N_h particles
if halo == 1:
    phi = np.random.uniform(0,2*np.pi, size = N_h)
    theta = np.arccos( np.random.uniform(-1,1, size = N_h) )
    r = e * np.sqrt( np.random.rand(N_h)**(-2.0/3.0)-1 )

    xh = r*np.sin(theta)*np.cos(phi)
    yh = r*np.sin(theta)*np.sin(phi)
    zh = r*np.cos(theta)

    halo_r = []
    for x,y,z in zip(xh,yh,zh):
        halo_r.append( (x,y,z) )

# Calculate the velocity (central + dispersion) of the disk particles
def Omega(r):
    B = a + b
    return G*M_d/(r**2 + B**2)**(3.0/2.0)

def Kappa(r):
    B = a + b
    num = G*M_d*(r**2+4*B**2)
    den = (r**2+B**2)**(5.0/2.0)
    return num/den

def Nu(r):
    B = a + b
    den = b*(r**2+B**2)**(3.0/2.0)
    return G*M_d*B/den

# 1) Calculate dispersion \sigma_R, \sigma_\theta, \sigma_z
def Sigma(R, M, a, b):
    B = a + b
    return (M * b**2) / (4 * np.pi) * (a * R**2 + (a + 3 * B) * (a + B)**2)\
         / (B**3 * (R**2 + (a + B)**2)**(5/2))

def sigma_R(r):
    Q0 = 1.5
    # Q_R = Q0 + spiral*0.5*np.tanh((r-5*a)/a)
    Q_R = Q0 + (0.4*np.exp(-(r/0.5/a)**2) + 0.4*(r>3*a)) * spiral
    return 3.36*G*Sigma(r,M_d,a,b)*Q_R/np.sqrt(Kappa(r))

def sigma_theta(r):
    return sigma_R(r)*np.sqrt(Kappa(r)/(2*Omega(r)))

def sigma_z(r):
    return sigma_R(r)*np.sqrt(Nu(r)/Kappa(r))

# 2) Combine everything together
def vel(r):
    return np.sqrt(r**2*Omega(r) + G*M_b*r**2/(r**2+c**2)**1.5\
         + halo*G*M_h*r**2/(r**2+e**2)**1.5)

r_d = np.sqrt(np.sum(disk[:,0:2]**2, axis=1))

rng = np.random.default_rng()
v_R = rng.normal(loc=0.0, scale=sigma_R(r_d)**0.5)
v_the = rng.normal(loc=vel(r_d), scale=sigma_theta(r_d)**0.5)
vz = rng.normal(loc=0.0, scale=sigma_z(r_d)**0.5)

vx = v_R * disk[:,0]/r_d - v_the * disk[:,1]/r_d
vy = v_R * disk[:,1]/r_d + v_the * disk[:,0]/r_d

v_disk = np.vstack((vx, vy, vz)).T

# Generate the velocity of the bulge particle
def sample_velocity(N,v_esc):
    samples = []
    while len(samples) < N:
        q = np.random.rand()  # trial q in [0,1]
        gq = q**2 * (1 - q**2)**(7/2)
        v_theta = np.arccos(np.random.uniform(-1,1))
        v_phi = np.random.uniform(0,2*np.pi)
        if np.random.rand() < gq/0.1:
            vs = v_esc[len(samples)] * q
            ths = v_theta
            phs = v_phi

            vxs = vs * np.sin(ths) * np.cos(phs)
            vys = vs * np.sin(ths) * np.sin(phs)
            vzs = vs * np.cos(ths)
            samples.append( (vxs,vys,vzs) )
    return np.array(samples[:N])
        
vec = (2*G*M_b)**0.5 * (r**2 + c**2)**(-0.25)
v_bulge = sample_velocity(N_b,vec)

# Generate the velocity of the halo particle
if halo == 1:
    vec = (2*G*M_h)**0.5 * (r**2 + e**2)**(-0.25)
    v_halo = sample_velocity(N_h,vec)

# Copy the first galaxy to generate the second galaxy
disk2 = disk.copy()
v_disk2 = v_disk.copy()
disk2[:,0] = disk2[:,0]*np.cos(angle) - disk[:,2]*np.sin(angle) + distance
disk2[:,2] = disk2[:,0]*np.sin(angle) + disk[:,2]*np.cos(angle)
disk[:,0]  = disk[:,0] - distance
v_disk2[:,0] = v_disk2[:,0]*np.cos(angle) - v_disk[:,2]*np.sin(angle)
v_disk2[:,2] = v_disk2[:,0]*np.sin(angle) + v_disk[:,2]*np.cos(angle)

bulge2 = bulge.copy()
v_bulge2 = v_bulge.copy()
bulge2[:] = bulge2[:] + np.array((distance,0,0))
bulge[:]  = bulge[:] - np.array((distance,0,0))
if halo == 1:
    halo_r2 = halo_r.copy()
    v_halo2 = v_halo.copy()
    halo_r2[:] = halo_r2[:] + np.array((distance,0,0))
    halo_r[:]  = halo_r[:] - np.array((distance,0,0))


# Print the result formally
if halo == 1: N = int(5*N)

print(2*N)
for i in range(2*N):
    print(m)
for i,j,k in disk:
    print(i,j,k)
for i,j,k in disk2:
    print(i,j,k)
for i,j,k in bulge:
    print(i,j,k)
for i,j,k in bulge2:
    print(i,j,k)
if halo == 1:
    for i,j,k in halo_r:
        print(i,j,k)
    for i,j,k in halo_r2:
        print(i,j,k)
print(B)
for i,j,k in v_disk:
    print(i,j,k)
for i,j,k in v_disk2:
    print(i,j,k)
for i,j,k in v_bulge:
    print(i,j,k)
for i,j,k in v_bulge2:
    print(i,j,k)
if halo == 1:
    for i,j,k in v_halo:
        print(i,j,k)
    for i,j,k in v_halo2:
        print(i,j,k)