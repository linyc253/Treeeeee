### You need to modify 'python' to your default python executable
PYTHON=/home/linyc253/.conda/envs/env_1/bin/python

### Generate data (modfy N below)
$PYTHON Initial_generator.py -N 500000 > Initial.dat

### Run
    cat > Input_Parameter.ini<<!
[BasicSetting]
DIM = 3                      # Dimension of the system (default: 3)
METHOD = 2                   # Method 1:brute_force 
                             #        2:tree_algo (default)
PARTICLE_FILE = Initial.dat  # filename of particle file (default: Initial.dat)
T_TOT = 100                   # total evolution time
DT = 0.1                    # maximal time interval
ETA = 0.05                    # parameter that controls the accuracy and stability of the timestep in simulations
TIME_PER_OUT = 2           # Output 00xxx.dat in every STEP_PER_OUT steps
EPSILON = 0.004              # softening length used to prevent singularities and numerical instabilities in particle interactions
OUTDIR = DATA

[Tree]
THETA = 0.4                  # Critical angle
POLES = 1
NCRIT = 1000

[Openmp]
THREADS = 4
CHUNK = 1

[GPU]
threadsPerBlock = 128
!
../../bin/treeeeee

$PYTHON plot_density.py
$PYTHON plot_energy.py