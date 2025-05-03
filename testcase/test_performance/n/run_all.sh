### You need to modify 'python' to your default python executable
PYTHON=python

echo "N   t1  t2" > comparison.dat

# Run METHOD = 1 & 2 for various N
for N in 1000 2000 4000 8000 16000 32000 64000
do
# Generate data
$PYTHON generate.py -N $N > Initial.dat


# Run METHOD = 1
    cat > Input_Parameter.ini<<!
[BasicSetting]
DIM = 3                      # Dimension of the system (default: 3)
METHOD = 1                   # Method 1:brute_force 
                             #        2:tree_algo (default)
PARTICLE_FILE = Initial.dat  # filename of particle file (default: Initial.dat)
T_TOT = 0.01                 # total evolution time
DT = 0.01                    # maximal time interval
STEP_PER_OUT = 3

[Tree]
THETA = 0.5               # Critical angle
!
../../../bin/treeeeee > log
echo  -n "$N  " >> comparison.dat
$PYTHON parse_time.py >> comparison.dat

# Run METHOD = 2
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
THETA = 0.5               # Critical angle
POLES = 2
NCRIT = 1

[Openmp]
THREADS = 8
CHUNK = 2
!
../../../bin/treeeeee > log
$PYTHON parse_time.py >> comparison.dat
echo "" >> comparison.dat


done

# Plot error vs THETA
$PYTHON plot.py
