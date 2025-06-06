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
T_TOT = 50                   # total evolution time
DT = 0.1                     # maximal time interval
ETA = 10.0                   # parameter that controls the accuracy and stability of the timestep in simulations
EPSILON = 1e-3               # softening length used to prevent singularities and numerical instabilities in particle interactions
TIME_PER_OUT = 0.5           # Output 00xxx.dat in every \Delta t = TIME_PER_OUT
OUTDIR = data                # where the output data stored
#RESTART = 12                 # restart from 00012.dat

[Tree]
THETA = 0.5                  # Critical angle
POLES = 2                    # 1: dipole (centre of mass)
                             # 2: quadrupole (3 pseudo-particles)
NCRIT = 1000                 # The max number of particles in a group (for constructing interaction list)

[Openmp]
THREADS = 4                  # Number of threads
CHUNK = 1                    # The chunk size in dynamic scheduling

[GPU]
threadsPerBlock = 128
!
../../../bin/treeeeee > log

$PYTHON parse_time.py >> comparison-$NAME.dat
done

echo "" >> comparison-$NAME.dat
done

# Plot error vs THETA
$PYTHON plot.py --suffix $NAME

