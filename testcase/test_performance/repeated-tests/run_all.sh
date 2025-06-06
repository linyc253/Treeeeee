### You need to modify 'python' to your default python executable
PYTHON=python
NAME=$1
REPEAT=$2

echo "N   time" > comparison-$NAME.dat

# Run METHOD = 1 & 2 for various N\
for N in 1000 2000 4000 8000 16000 32000
do
echo  -n "$N  " >> comparison-$NAME.dat

for ((i=0; i<REPEAT; ++i));
do
# Generate data
$PYTHON generate.py -N $N > Initial.dat

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
CHUNK = 32
!
../../../bin/treeeeee > log

$PYTHON parse_time.py >> comparison-$NAME.dat
done

echo "" >> comparison-$NAME.dat
done

# Plot error vs THETA
$PYTHON plot.py --suffix $NAME

