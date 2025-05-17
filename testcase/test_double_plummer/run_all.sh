### You need to modify 'python' to your default python executable
PYTHON=/home/linyc253/.conda/envs/env_1/bin/python

### Generate data (modfy N below)
# N is the particle number for "each" plummer
$PYTHON Initial_generator.py -N 50000 > Initial.dat

### Run METHOD = 2
    cat > Input_Parameter.ini<<!
[BasicSetting]
DIM = 3                      # Dimension of the system (default: 3)
METHOD = 2                   # Method 1:brute_force 
                             #        2:tree_algo (default)
PARTICLE_FILE = Initial.dat  # filename of particle file (default: Initial.dat)
T_TOT = 10                   # total evolution time
DT = 0.03                    # maximal time interval
ETA = 5.0                    # parameter that controls the accuracy and stability of the timestep in simulations
TIME_PER_OUT = 0.5           # Output 00xxx.dat in every STEP_PER_OUT steps
EPSILON = 1e-4               # softening length used to prevent singularities and numerical instabilities in particle interactions
OUTDIR = DATA

[Tree]
THETA = 0.4                  # Critical angle
POLES = 1
NCRIT = 1000

[Openmp]
THREADS = 2
CHUNK = 1
!
../../bin/treeeeee > log

# Plot plummer animation
cd Figure
$PYTHON plot_gas.py -N 401
cd ..
convert Figure/0*.png Plummer2.gif
#$PYTHON Energy.py -N 201