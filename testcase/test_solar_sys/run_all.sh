### You need to modify 'python' to your default python executable
PYTHON=python3

### Generate data (modfy N below)
$PYTHON Initial_generator.py -N 8 > Initial.dat

### Run METHOD = 2
    cat > Input_Parameter.ini<<!
[BasicSetting]
DIM = 3                      # Dimension of the system (default: 3)
METHOD = 2                   # Method 1:brute_force 
                             #        2:tree_algo (default)
PARTICLE_FILE = Initial.dat  # filename of particle file (default: Initial.dat)
T_TOT = 30                   # total evolution time
DT = 0.02                     # maximal time interval
STEP_PER_OUT = 3             # steps per data output

[Tree]
THETA = 0.4                  # Critical angle

!
../../bin/treeeeee

# Plot plummer animation
$PYTHON Energy.py
$PYTHON plot.py
convert 00*.png Solar_sys.gif