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
cd testcase/test_IO
../../bin/treeeeee
```
# Parameter file format
The parameter file should be named as `Input_Parameter.ini`. Use `#` for comment.
```
[BasicSetting]
DIM = 2                      # Dimension of the system (default: 3)
METHOD = 1                   # Method 1:brute_force 
                             #        2:tree_algo (default)
PARTICLE_FILE = Initial.dat  # filename of particle file (default: Initial.dat)
T_TOT = 5.1                  # total evolution time
DT = 0.2                     # maximal time interval

[Tree]
THETA = 0.003                # Critical angle
```
# Particle file format
The particle file (name should be specified in 'parameter file') must follow the format below
```
npart
mass(1)
...
mass(npart)
x(1),y(1),z(1)
...
x(npart),y(npart),z(npart)
vx(1),vy(1),vz(1)
...
vx(npart),vy(npart),vz(npart)
```