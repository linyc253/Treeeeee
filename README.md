# Quick Start
First, clone by the following command
```
git clone https://github.com/linyc253/Treeeeee.git
```
Then compile the code by
```
cd Treeeeee
make
```
And execute by
```
cd testcase/test_random
python generate.py > Initial.dat
../../bin/treeeeee
```
# Parameter file format
The parameter file should be named as `Input_Parameter.ini`. Use `#` for comment.
```
[BasicSetting]
DIM = 3                      # Dimension of the system (default: 3)
METHOD = 2                   # Method 1:brute_force 
                             #        2:tree_algo (default)
PARTICLE_FILE = Initial.dat  # filename of particle file (default: Initial.dat)
T_TOT = 0.1                  # total evolution time
DT = 0.005                   # maximal time interval
ETA = 1.0                    # parameter that controls the accuracy and stability of the timestep in simulations
EPSILON = 1e-4               # softening length used to prevent singularities and numerical instabilities in particle interactions
TIME_PER_OUT = 0.02          # Output 00xxx.dat in every \Delta t = TIME_PER_OUT

[Tree]
THETA = 0.5                  # Critical angle
POLES = 1                    # 1: dipole (centre of mass)
                             # 2: quadrupole (3 pseudo-particles)
```
# Particle file format
The particle file (filename should be specified in `Input_Parameter.ini`) must follow the format below
```
n_particles                                   # number of particles
mass(1)                                       # n_particles lines of masses
...
mass(n_particles)
x(1),y(1),z(1)                                # n_particles lines of initial positions
...
x(n_particles),y(n_particles),z(n_particles)
vx(1),vy(1),vz(1)                             # n_particles lines of initial velocities
...
vx(n_particles),vy(n_particles),vz(n_particles)
```