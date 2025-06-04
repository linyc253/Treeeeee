### You need to modify 'python' to your default python executable
PYTHON=/home/linyc253/.conda/envs/env_1/bin/python

### Generate data (modfy N below)
# N is the particle number for "each" plummer
$PYTHON Initial_generator.py -N 500000 -V 1 > Initial.dat

### Run
    cat > Input_Parameter.ini<<!
[BasicSetting]
DIM = 3                      # Dimension of the system (default: 3)
METHOD = 2                   # Method 1:brute_force 
                             #        2:tree_algo (default)
PARTICLE_FILE = Initial.dat  # filename of particle file (default: Initial.dat)
T_TOT = 200                   # total evolution time
DT = 0.1                    # maximal time interval
ETA = 0.05                    # parameter that controls the accuracy and stability of the timestep in simulations
TIME_PER_OUT = 0.5           # Output 00xxx.dat in every STEP_PER_OUT steps
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

$PYTHON plot_energy.py

# Plot plummer animation
cd Figure
$PYTHON plot_gas.py -N 400
cd ..
ffmpeg -framerate 12 -i Figure/%05d.png plummer2.mp4
