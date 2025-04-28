import numpy as np
import random

# Parameters
G = 1           # Newton's gravity constant
N = 5           # number of planets
M = 30000       # mass of the sun

# Generate masses of N planets
m = np.zeros(N+1)
for i in range(N):
    m[i] = np.random.uniform(1, 100.0)
m[-1] = M

# Generate position of N planets
x = np.zeros(N+1)         # set up the Cartesian coordinate 
y = np.zeros(N+1)
z = np.zeros(N+1)

for i in range(N):
    x[i] = np.random.uniform(1, 30.0)

# Generate the velocity of N planets
vx = np.zeros(N+1)        # Cartesian coordinate in velocity space
vy = np.zeros(N+1)
vz = np.zeros(N+1)

for i in range(N):
    # calculate the speed by energy conservation
    vy[i] = (G*M**2/(M+m[i])/x[i])**0.5 * (1 + 0.2 * np.random.uniform(0, 1))
    
# Print the result formally
print(N+1)
for i in range(N+1):
    print(m[i])
for i in range(N+1):
    print('%12f %13f %14f' % (x[i], y[i], z[i]))
for i in range(N+1):
    print('%15f %16f %17f' % (vx[i], vy[i], vz[i]))