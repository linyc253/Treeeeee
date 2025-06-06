### You need to modify 'python' to your default python executable
PYTHON=/home/linyc253/.conda/envs/env_1/bin/python

### Generate data (modfy N below)
# N is the particle number for "each" plummer
$PYTHON Initial_generator.py -N 20000 -V 0 > Initial.dat

### Run non-rotation double plummer
    cat > Input_Parameter.ini<<!
[BasicSetting]
DIM = 3                      # Dimension of the system (default: 3)
METHOD = 2                   # Method 1:brute_force 
                             #        2:tree_algo (default)
PARTICLE_FILE = Initial.dat  # filename of particle file (default: Initial.dat)
T_TOT = 200                  # total evolution time
DT = 0.3                     # maximal time interval
ETA = 5.0                    # parameter that controls the accuracy and stability of the timestep in simulations
TIME_PER_OUT = 0.5           # Output 00xxx.dat in every STEP_PER_OUT steps
EPSILON = 1e-5               # softening length used to prevent singularities and numerical instabilities in particle interactions
OUTDIR = DATA1

[Tree]
THETA = 0.4                  # Critical angle
POLES = 1
NCRIT = 1200

[Openmp]
THREADS = 4
CHUNK = 1
!
../../bin/treeeeee > log

# Plot plummer animation
mv Energy.dat Figure/Energy1.dat

### Generate data
$PYTHON Initial_generator.py -N 20000 -V 1 > Initial2.dat

### Run rotation double plummer
    cat > Input_Parameter.ini<<!
[BasicSetting]
DIM = 3                      # Dimension of the system (default: 3)
METHOD = 2                   # Method 1:brute_force 
                             #        2:tree_algo (default)
PARTICLE_FILE = Initial2.dat  # filename of particle file (default: Initial.dat)
T_TOT = 200                  # total evolution time
DT = 0.3                     # maximal time interval
ETA = 5.0                    # parameter that controls the accuracy and stability of the timestep in simulations
TIME_PER_OUT = 0.5           # Output 00xxx.dat in every STEP_PER_OUT steps
EPSILON = 1e-5               # softening length used to prevent singularities and numerical instabilities in particle interactions
OUTDIR = DATA2

[Tree]
THETA = 0.4                  # Critical angle
POLES = 1
NCRIT = 1200

[Openmp]
THREADS = 4
CHUNK = 1
!
../../bin/treeeeee > log2

# Plot plummer animation
mv Energy.dat Figure/Energy2.dat

cd Figure
$PYTHON energy.py
$PYTHON plot.py -F 400
cd ..

