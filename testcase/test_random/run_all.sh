### You need to modify 'python' to your default python executable
PYTHON=python3

# Generate data
$PYTHON generate.py -N 10000 > Initial.dat

### Run METHOD = 2
    cat > Input_Parameter.ini<<!
[BasicSetting]
DIM = 3                      # Dimension of the system (default: 3)
METHOD = 2                   # Method 1:brute_force 
                             #        2:tree_algo (default)
PARTICLE_FILE = Initial.dat  # filename of particle file (default: Initial.dat)
T_TOT = 100                  # total evolution time
DT = 0.005                   # maximal time interval
ETA = 1.0                    # parameter that controls the accuracy and stability of the timestep in simulations
EPSILON = 1e-4               # softening length used to prevent singularities and numerical instabilities in particle interactions
TIME_PER_OUT = 0.4           # Output 00xxx.dat in every \Delta t = TIME_PER_OUT

[Tree]
THETA = 0.5                  # Critical angle
POLES = 1                    # 1: dipole (centre of mass)
                             # 2: quadrupole (3 pseudo-particles)

!
../../bin/treeeeee

# Plot plummer animation
$PYTHON plot_gas.py
convert 000*.png Random.gif