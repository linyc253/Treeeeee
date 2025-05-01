import numpy as np
import matplotlib.pyplot as plt
import argparse

# Parameter
G = 1                   # Newton gravity constant
N = 2


# Read parameter from script
# parser = argparse.ArgumentParser()
# parser.add_argument("-N", "--N_f", help = "Number of figures")
# args = parser.parse_args()
# N_f = int(args.N)

N_f = 126

parser = argparse.ArgumentParser()
parser.add_argument("-D", "--N", help = "Distance")
args = parser.parse_args()
D = float(args.N)

x = np.empty((N_f,N,3))
v = np.empty((N_f,N,3))

# Load the file
for i in range(N_f):
    filename = "%05d" % i
    # Load the file
    with open(filename+".dat", 'r') as f:
        lines = f.readlines()

    # Masses:
    m = np.array([float(lines[i].strip()) for i in range(1, 1+N)])

    # Positions:
    pos = np.array([
        list(map(float, lines[i].strip().split()))
        for i in range(1+N, 1+2*N)
    ])

    # Velocities:
    vel = np.array([
        list(map(float, lines[i].strip().split()))
        for i in range(1+2*N, 1+3*N)
    ])
    for j in range(2):
        x[i,j,:] = pos[j,:]
        v[i,j,:] = vel[j,:]

E_k = np.zeros(N_f-1)
U_g = np.zeros(N_f-1)
E_t = np.zeros(N_f-1)
for i in range(1, N_f):
    # kinetic energy
    E_k[i-1] = 0.5 * np.sum(m * np.sum(v[i]**2, axis=1))
    
    # potential energy
    dx = x[i,:,np.newaxis,0] - x[i,np.newaxis,:,0]
    dy = x[i,:,np.newaxis,1] - x[i,np.newaxis,:,1]
    dz = x[i,:,np.newaxis,2] - x[i,np.newaxis,:,2]
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    
    m_outer = m[:,np.newaxis] * m[np.newaxis,:]
    
    upper = np.triu_indices(N, k=1)
    U_g[i-1] = -G * np.sum(m_outer[upper] / r[upper])
    
    # total energy
    E_t[i-1] = E_k[i-1] + U_g[i-1]


# Create plot
plt.figure()
plt.plot(E_t)
plt.title("Total Energy")
plt.xlabel("Time")
plt.ylabel("Energy")
# plt.ylim(bottom=0, top=E_t.max()*1.1)
plt.savefig(f"Tot_E/Solar_Energy_{D:.2f}.png")

plt.figure()
plt.plot(E_k)
plt.title("Kinetic Energy")
plt.xlabel("Time")
plt.ylabel("Kinetic")
plt.savefig(f"E_K/Solar_Kinetic_{D:.2f}.png")

plt.figure()
plt.plot(U_g)
plt.title("Potential Energy")
plt.xlabel("Time")
plt.ylabel("Potential")
plt.savefig(f"U_g/Solar_Potential_{D:.2f}.png")