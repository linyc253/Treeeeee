### Some parameter need to be tuned:
# PYTHON = python / python3 (according to your version) in line 8
# N for "Initial_generator.py" in line 11
# Setting between line 21-26.
# file name for "mv Final.dat". The new name should be labeled in line 32.
# N_f for "plot_gas.py" in line 35. This should be "Final.dat"+2 

### You need to modify 'python' to your default python executable
PYTHON=python3

### Generate data (modfy N below)
$PYTHON generate.py -N 10000 > Initial.dat

### Run METHOD = 2
    cat > Input_Parameter.ini<<!
[BasicSetting]
DIM = 3                      # Dimension of the system (default: 3)
METHOD = 2                   # Method 1:brute_force 
                             #        2:tree_algo (default)
PARTICLE_FILE = Initial.dat  # filename of particle file (default: Initial.dat)
T_TOT = 10                   # total evolution time
DT = 0.1                     # maximal time interval
STEP_PER_OUT = 3             # steps per data output

[Tree]
THETA = 0.4                  # Critical angle

!
../../bin/treeeeee

### rename "Final.dat" (this number is n=[T_TOT/(DT*STEP_PER_OUT)]+1)
mv Final.dat 00034.dat

### Plot plummer animation (N_f = n+1)
$PYTHON plot_gas.py -N_f 35
convert 000*.png Random.gif