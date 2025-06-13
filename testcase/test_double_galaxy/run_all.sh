### You need to modify 'python' to your default python executable
PYTHON=python

### Generate data ( for H=1, N_tot = N*(5R+1) )
$PYTHON Initial_generator.py -N 200000 -S 0.3 -R 0.9 -H 0 > Initial.dat
$PYTHON plot.py -R 0.9

### Run Treeeeee setting
    cat > Input_Parameter.ini<<!
[BasicSetting]
DIM = 3                      # Dimension of the system (default: 3)
METHOD = 2                   # Method 1:brute_force 
                             #        2:tree_algo (default)
PARTICLE_FILE = Initial.dat  # filename of particle file (default: Initial.dat)
T_TOT = 4000                 # total evolution time
DT = 1.0                     # maximal time interval
ETA = 20.0                   # parameter that controls the accuracy and stability of the timestep in simulations
TIME_PER_OUT = 4             # Output 00xxx.dat in every STEP_PER_OUT steps
EPSILON = 1e-3               # softening length used to prevent singularities and numerical instabilities in particle interactions
OUTDIR = DATA                # The output file should be put in this folder
# RESTART = 1000                # Continue the calculation at this step

[Tree]
THETA = 0.4                  # Critical angle
POLES = 1
NCRIT = 1200

[Openmp]
THREADS = 2
CHUNK = 1
!
../../bin/treeeeee > log

# Plot galaxy animation
$PYTHON plot_energy.py

cd Figure
$PYTHON plot_gas.py -F 1000 -R 0.9 -H 0 &
$PYTHON plot_gas2.py -F 1000 -R 0.9 -H 0 &
cd ..
wait
ffmpeg -framerate 24 -i Figure/%05d.png galaxy2.mp4