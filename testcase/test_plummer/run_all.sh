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
T_TOT = 3000                   # total evolution time
DT = 10                     # maximal time interval
STEP_PER_OUT = 3             # steps per data output

[Tree]
THETA = 0.4                  # Critical angle

!
../../bin/treeeeee

# Plot plummer animation
$PYTHON plot_gas.py
convert 000*.png Plummer.gif