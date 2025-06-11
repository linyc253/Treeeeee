### You need to modify 'python' to your default python executable
PYTHON=python

# Setting
    cat > Input_Parameter.ini<<!
[BasicSetting]
DIM = 3                      # Dimension of the system (default: 3)
METHOD = 1                   # Method 1:brute_force 
                             #        2:tree_algo (default)
PARTICLE_FILE = Initial.dat  # filename of particle file (default: Initial.dat)
T_TOT = 10                 # total evolution time
DT = 0.01                    # maximal time interval
STEP_PER_OUT = 8

[Tree]
!

# Generate data
for DISTANCE in 10.00 8.00 6.00 4.00 3.00 2.00 1.00 0.50 0.25
do
$PYTHON Initial_generator.py -D $DISTANCE > Initial.dat

../../bin/treeeeee
mv Initial.dat 00000.dat
$PYTHON plot.py -D $DISTANCE
$PYTHON Energy.py -D $DISTANCE
convert 00*.png distance_$DISTANCE.gif
sh remove.sh
done
