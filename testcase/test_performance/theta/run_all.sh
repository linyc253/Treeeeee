### You need to modify 'python' to your default python executable
PYTHON=python

# Generate data
$PYTHON generate.py -N 10000 > Initial.dat
echo "THETA    error" > comparison.dat

# Run METHOD = 2 for various THETA
for THETA in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    cat > Input_Parameter.ini<<!
[BasicSetting]
DIM = 3                      # Dimension of the system (default: 3)
METHOD = 2                   # Method 1:brute_force 
                             #        2:tree_algo (default)
PARTICLE_FILE = Initial.dat  # filename of particle file (default: Initial.dat)
T_TOT = 0.01                 # total evolution time
DT = 0.01                    # maximal time interval
STEP_PER_OUT = 3

[Tree]
THETA = $THETA               # Critical angle
!
../../../bin/treeeeee > log
echo  -n "$THETA  " >> comparison.dat
$PYTHON parse_time.py >> comparison.dat
done

# Plot error vs THETA
$PYTHON plot.py
