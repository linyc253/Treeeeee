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
alternatively, use CMake
```
cd Treeeeee
cmake -S . -B build && cmake --build build
```
To clean all binaries and CMake files:
```
cmake -build build --target clean-all
```
additional cmake options can be turned on, "-DOMP" for OpenMP, "-DCUDA" for cuda support, and "-DDEBUG" for some additional timing information. For example, 
```
cd Treeeeee
cmake -DOMP=ON -DCUDA=ON -S . -B build && cmake --build build
```
Once compiled you can execute by
```
cd testcase/test_random
python generate.py -N 10000 > Initial.dat
../../bin/treeeeee
```
The calculation result can be visualized by the python script
```
python plot_gas.py
convert data/*.png Random.gif
```
# Parameter file format
The parameter file should be named as `Input_Parameter.ini`. Use `#` for comment.
```
[BasicSetting]
DIM = 3                      # Dimension of the system (default: 3)
METHOD = 2                   # Method 1:brute_force 
                             #        2:tree_algo (default)
PARTICLE_FILE = Initial.dat  # filename of particle file (default: Initial.dat)
T_TOT = 50.0                 # total evolution time
DT = 0.1                     # maximal time interval
ETA = 10.0                   # parameter that controls the accuracy and stability of the timestep in simulations
EPSILON = 1e-3               # softening length used to prevent singularities and numerical instabilities in particle interactions
TIME_PER_OUT = 0.5           # Output 00xxx.dat in every \Delta t = TIME_PER_OUT
OUTDIR = data                # where the output data stored
#RESTART = 12                 # restart from 00012.dat

[Tree]
THETA = 0.5                  # Critical angle
POLES = 1                    # 1: dipole (centre of mass)
                             # 2: quadrupole (3 pseudo-particles)
NCRIT = 1000                 # The max number of particles in a group (for constructing interaction list)

[Openmp]
THREADS = 4                  # Number of threads
CHUNK = 1                    # The chunk size in dynamic scheduling

[GPU]
threadsPerBlock = 128
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
