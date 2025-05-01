import numpy as np
import matplotlib.pyplot as plt
import argparse

# Parameter
G = 1           # Newton gravity constant

parser = argparse.ArgumentParser()
parser.add_argument("-N", "--N_f", help = "Number of figures")
args = parser.parse_args()
N_f = int(args.N_f)

# array
filename = "Initial.dat"
with open(filename) as f:
    lines = f.readlines()
N = int(lines[0].strip())
x = np.empty((N_f,N,3))
v = np.empty((N_f,N,3))

# Load the file
for i in range(1, N_f):
    filename = "%05d" % i
    # Load the file
    with open(filename+".dat", 'r') as f:
        lines = f.readlines()

    # Masses:
    m = np.array([float(lines[i].strip()) for i in range(1, 1 + N)])

    # Positions:
    pos = np.array([
        list(map(float, lines[i].strip().split()))
        for i in range(1 + N, 1 + 2 * N)
    ])

    # Velocities:
    vel = np.array([
        list(map(float, lines[i].strip().split()))
        for i in range(1 + 2 * N, 1 + 3 * N)
    ])
    for j in range(N):
        x[i,j,:] = pos[j,:]
        v[i,j,:] = vel[j,:]

E_k = np.zeros(N_f-1)
U_g = np.zeros(N_f-1)
E_t = np.zeros(N_f-1)
for i in range(1,N_f):
    for j in range(N-1):
        E_k[i-1] += 0.5 * m[j] * (v[i,j,0]**2 + v[i,j,1]**2 + v[i,j,2]**2)
        for k in range(j+1,N):
            U_g[i-1] -= G*m[j]*m[k]/((x[i,j,0]-x[i,k,0])**2 + (x[i,j,1]-x[i,k,1])**2 +\
                                 (x[i,j,2]-x[i,k,2])**2)**0.5
    E_k[i-1] += 0.5 * m[N-1] * (v[i,N-1,0]**2 + v[i,N-1,1]**2 + v[i,N-1,2]**2)
    E_t[i-1] = E_k[i-1] + U_g[i-1]

# Create plot
plt.figure()
plt.plot(E_t)
plt.title("Total Energy")
plt.xlabel("Time")
plt.ylabel("Energy")
# plt.ylim(bottom=0, top=E_t.max()*1.1)
plt.savefig("Solar_Energy.png")

plt.figure()
plt.plot(E_k)
plt.title("Kinetic Energy")
plt.xlabel("Time")
plt.ylabel("Kinetic")
plt.savefig("Solar_Kinetic.png")

plt.figure()
plt.plot(U_g)
plt.title("Potential Energy")
plt.xlabel("Time")
plt.ylabel("Potential")
plt.savefig("Solar_Potential.png")