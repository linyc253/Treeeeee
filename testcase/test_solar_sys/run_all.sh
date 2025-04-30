### You need to modify 'python' to your default python executable
PYTHON=python3

### Generate data (modfy N below)
#$PYTHON Initial_generator.py -N 10 > Initial.dat
$PYTHON random_generator.py -N 10 > Initial.dat

### Setting
    cat > Input_Parameter.ini<<!
[BasicSetting]
DIM = 3                      # Dimension of the system (default: 3)
METHOD = 1                   # Method 1:brute_force 
                             #        2:tree_algo (default)
PARTICLE_FILE = Initial.dat  # filename of particle file (default: Initial.dat)
T_TOT = 10                   # total evolution time
DT = 0.0001                    # maximal time interval
STEP_PER_OUT = 200             # steps per data output

[Tree]
THETA = 0.4                  # Critical angle

!
../../bin/treeeeee

### Plot plummer animation (N = T_TOT/(DT*STEP_PER_OUT)+1)
$PYTHON Energy.py -N 501
$PYTHON plot.py -N 501
convert 0*.png Solar_sys.gif