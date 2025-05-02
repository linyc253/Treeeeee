### You need to modify 'python' to your default python executable
PYTHON=python3

### Generate data (modfy N below)
$PYTHON Initial_generator.py -N 10000 > Initial.dat

### Run METHOD = 2
    cat > Input_Parameter.ini<<!
[BasicSetting]
DIM = 3                      # Dimension of the system (default: 3)
METHOD = 2                   # Method 1:brute_force 
                             #        2:tree_algo (default)
PARTICLE_FILE = Initial.dat  # filename of particle file (default: Initial.dat)
T_TOT = 10                   # total evolution time
DT = 0.04                    # maximal time interval
ETA = 1.0                    # parameter that controls the accuracy and stability of the timestep in simulations
TIME_PER_OUT = 0.2           # Output 00xxx.dat in every STEP_PER_OUT steps
EPSILON = 1e-4               # softening length used to prevent singularities and numerical instabilities in particle interactions

[Tree]
THETA = 0.3                  # Critical angle
POLES = 1
!
../../bin/treeeeee

# Plot plummer animation
$PYTHON plot_gas.py -N 51
convert 000*.png Plummer.gif
$PYTHON Energy.py -N 51